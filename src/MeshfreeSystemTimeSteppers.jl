
# Default case: Everything is decoupled
export RelaxationStepper, ARS2IMEX, GeneralIMEXTimeStepper, ARS233, PareschiRussoIMEXSSP3, ARS222, SSP2332, SimpleSplitting, RalstonRK2, ARS232

include("ButcherTableaus.jl")

function (ts::TimeStepper)(eqs::Vector{<:ScalarHyperbolicEquation}, particleGrids::Vector{<:ParticleGrid}, settings::SimSetting, time::Real, dt::Real)
    @warn "No dedicated system timestepper found. System will be treated independently using the given scalar timestepper."
    for (i,particleGrid) = enumerate(particleGrids)
        ts(eqs[i], particleGrid, settings, time, dt)
    end
end

function (ts::TimeStepper)(eq::ScalarHyperbolicEquation, particleGrids::Vector{<:ParticleGrid}, settings::SimSetting, time::Real, dt::Real)
    @warn "Only one scalar hyperbolic equation for a system is found. The scalar equation will be used for all components!"
    ts([eq for _ = eachindex(particleGrids)], particleGrids, settings, time, dt)
end

struct RelaxationStepper <: MeshfreeSystemTimeStepper
    timestepper::TimeStepper
    M::Vector{T} where T <: Function
    epsilon::Float64
    rho::Vector{Float64}

    function RelaxationStepper(timestepper::TimeStepper, Nx::Int64, M::Vector{T}; epsilon = 10. ^-10) where T <: Function
        @assert !isa(timestepper, MeshfreeSystemTimeStepper) "A scalar timestepper has to be given!"
        new(timestepper, M, epsilon, Vector{Float64}(undef, Nx))
    end
end

function initTimeStepper(method::RelaxationStepper, particleGrids::Vector{T}, settings::SimSetting) where T <: ParticleGrid
    initTimeStepper(method.timestepper, particleGrids[1], settings)
end

function(relax_ts::RelaxationStepper)(eqs::Vector{LinearAdvection{T}}, particleGrids::Vector{<:ParticleGrid}, settings::SimSetting, time::Real, dt::Real) where T <: Float64
    for (i,pg) = enumerate(particleGrids)
        relax_ts.timestepper(eqs[i], pg, settings, time, dt)
    end
    relax_ts.rho[:] = sum([[p.rho for p = pg.grid] for pg = particleGrids])
    for (i_pg,pg) = enumerate(particleGrids)
        
        for (i_p,p) = enumerate(pg.grid)
            p.rho = relax_ts.epsilon/(relax_ts.epsilon + dt) * p.rho + dt/(relax_ts.epsilon+dt) * relax_ts.M[i_pg](relax_ts.rho[i_p])
        end
    end

end

# In your TimeIntegration.jl or a similar module

# Ensure all necessary types are accessible via `using` statements
# using ..ParticleGrids, ..SimSettings, ..ScalarHyperbolicEquations, ..SourceTerms

"""
    SimpleSplitting <: MeshfreeSystemTimeStepper

A simple first-order operator splitting time stepper for relaxation systems.
It performs an advection step followed by a relaxation step.

This is a robust but only first-order accurate method, often used for reference
or as a component in more complex schemes.
"""
struct SimpleSplitting{T <: TimeStepper, S <: AbstractSourceTerm} <: MeshfreeSystemTimeStepper
    timestepper::T # The scalar timestepper for the advection step (e.g., EulerUpwind, RalstonRK2)
    source_term::S # The RelaxationSourceTerm object

    # Buffer to hold the intermediate macroscopic state after advection
    macro_state_buffer::Matrix{Float64} 

    function SimpleSplitting(
        timestepper::T, 
        source_term::S,
        N_total_particles::Int
    ) where {T <: TimeStepper, S <: AbstractSourceTerm}
        @assert !isa(timestepper, MeshfreeSystemTimeStepper) "A scalar timestepper must be provided for the advection step."
        @assert isa(source_term, RelaxationSourceTerm) "Source term must be a RelaxationSourceTerm."
        
        num_macro_vars = source_term.num_macro_variables
        macro_buffer = Matrix{Float64}(undef, N_total_particles, num_macro_vars)
        
        new{T, S}(timestepper, source_term, macro_buffer)
    end
end

function initTimeStepper(method::SimpleSplitting, particleGrids::Vector{<:ParticleGrid}, settings::SimSetting)
    # Initialize the underlying scalar timestepper for each component grid
    # (This assumes the initTimeStepper for the scalar method is defined)
    for pg in particleGrids
        initTimeStepper(method.timestepper, pg, settings)
    end
end

"""
    (ss::SimpleSplitting)(eqs, particleGrids, settings, time, dt)

Functor for the SimpleSplitting timestepper.
"""
function (ss::SimpleSplitting)(
    eqs::Vector{<:LinearAdvection}, # The kinetic equations
    particleGrids::Vector{<:ParticleGrid}, 
    settings::SimSetting, 
    time::Real, 
    dt::Real
)
    # --- 1. Advection Step ---
    # Apply the scalar timestepper to each kinetic component grid.
    # This updates the .rho field of each particle to the post-advection state v_k^*.
    for k_comp in eachindex(particleGrids)
        ss.timestepper(eqs[k_comp], particleGrids[k_comp], settings, time, dt)
    end
    # Note: The scalar timestepper should only update interior points.

    # --- 2. Recombination Step ---
    # Reconstruct the macroscopic state U_macro^* = [rho^*, m^*, E^*] at ALL particle locations
    # (including ghosts) because Maxwellians for interior points may need neighbor values.
    N_total_particles = length(particleGrids[1].grid)
    rs = ss.source_term # Alias for the relaxation source term object

    for p_idx in 1:N_total_particles
        for i_macro in 1:rs.num_macro_variables
            # Sum the advected kinetic variables (v_k^*) to get the macroscopic state (U_macro_i^*)
            indices = rs.kinetic_indices[i_macro]
            ss.macro_state_buffer[p_idx, i_macro] = sum(particleGrids[k_idx].grid[p_idx].rho for k_idx in indices)
        end
    end

    # --- 3. Relaxation Step ---
    # Apply the relaxation formula ONLY to the INTERIOR particles.
    coeff_ep = rs.epsilon / (rs.epsilon + dt)
    coeff_dt = dt / (rs.epsilon + dt)

    for k_comp in eachindex(particleGrids)
        pg_k = particleGrids[k_comp]
        maxwellian_func_k = rs.maxwellians[k_comp]

        for p_idx in pg_k.interior_indices
            particle = pg_k.grid[p_idx]
            
            # This is v_k^*(p_idx) from the advection step
            v_k_star_at_p = particle.rho 
            
            # Get the macroscopic state vector (rho^*, m^*, E^*) at this particle
            U_macro_star_at_p = NTuple{rs.num_macro_variables, Float64}(ss.macro_state_buffer[p_idx, i] for i in 1:rs.num_macro_variables)

            # Evaluate the Maxwellian by splatting the macroscopic state tuple
            equilibrium_val_k = maxwellian_func_k(U_macro_star_at_p...)
            
            # Update particle.rho to the final state v_k^{n+1}
            particle.rho = coeff_ep * v_k_star_at_p + coeff_dt * equilibrium_val_k
        end
    end
    # Ghost cell values in particleGrids are NOT touched in this step, preserving the BCs.
end

# --- Helper to compute explicit tendency -L(U_state) with MOOD ---
function compute_explicit_tendency_with_mood!(
    K_E_out_sys::Matrix{Float64},
    U_state_sys::Matrix{Float64},
    gradientInterpolator::Interpolations.GradientInterpolator, # Added Interpolations.
    fallbackInterpolator::Union{Interpolations.GradientInterpolator, Nothing}, # Added Interpolations.
    mood_criterion::MOODCriterion, # Assuming MOODCriterion is defined and imported
    scalar_equations::Vector{<:ScalarHyperbolicEquations.ScalarHyperbolicEquation}, # Added ScalarHyperbolicEquations.
    component_grids::Vector{<:ParticleGrids.ParticleGrid}, # Added ParticleGrids.
    settings::SimSettings.SimSetting, # Added SimSettings.
    dt_for_mood_check::Real,
    is_first_mood_stage_in_rk::Bool,
    interior_indices::UnitRange{Int64}
)
    N_particles = size(U_state_sys, 1)
    N_components = length(scalar_equations)

    if N_components != size(U_state_sys, 2) || N_components != length(component_grids)
        error("Mismatch in number of components for equations, states, or grids in compute_explicit_tendency_with_mood!.")
    end

    for k_comp in 1:N_components
        scalar_grid_k = component_grids[k_comp]
        scalar_eq_k = scalar_equations[k_comp]
        U_state_k_view = @view U_state_sys[:, k_comp]

        if hasmethod(initTimeStep, (typeof(gradientInterpolator), typeof(scalar_grid_k), Real, Real, Vector{Float64}))
            initTimeStep(gradientInterpolator, scalar_grid_k, settings.interpAlpha, settings.interpRange)
        end
        if !isnothing(fallbackInterpolator) && hasmethod(initTimeStep, (typeof(fallbackInterpolator), typeof(scalar_grid_k), Real, Real, Vector{Float64}))
            initTimeStep(fallbackInterpolator, scalar_grid_k, interpAlpha, settings.interpRange)
        end
        
        temp_rho_backup_k = [p.rho for p in scalar_grid_k.grid]
        for p_idx_cv in interior_indices; scalar_grid_k.grid[p_idx_cv].rho = U_state_k_view[p_idx_cv]; end
        
        copyCurvatures!(scalar_grid_k) 

        for p_idx in interior_indices
            div_high_k_p = gradientInterpolator(scalar_grid_k, p_idx, U_state_k_view, scalar_eq_k, settings; setCurvature=true)
            rho_candidate_for_mood = U_state_k_view[p_idx] - dt_for_mood_check * div_high_k_p
            K_E_out_sys[p_idx, k_comp] = -div_high_k_p

            if !isnothing(fallbackInterpolator) && mood_criterion(scalar_grid_k, p_idx, U_state_k_view, rho_candidate_for_mood; firstStage=is_first_mood_stage_in_rk)
                div_fallback_k_p = fallbackInterpolator(
                    scalar_grid_k, p_idx, U_state_k_view, scalar_eq_k, settings; setCurvature=false
                )
                K_E_out_sys[p_idx, k_comp] = -div_fallback_k_p
            end
        end
        for p_idx_cv in interior_indices; scalar_grid_k.grid[p_idx_cv].rho = temp_rho_backup_k[p_idx_cv]; end
    end
end

# DEPRECATED, use general time stepper
# --- ARS2IMEX Time Stepper Struct ---
struct ARS2IMEX{
    G1 <: GradientInterpolator, # Added Interpolations.
    G2 <: Union{GradientInterpolator, Nothing}, # Added Interpolations.
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: AbstractImplicitSolver # Added ImplicitSolvers.
} <: TimeStepper # Added 

    gradientInterpolator::G1
    fallbackInterpolator::G2 
    mood::M
    implicit_solver::IS
    source_term::AbstractSourceTerm

    U_n_sys::Matrix{Float64}
    U_stage1_sys::Matrix{Float64} 
    U_stage2_sys::Matrix{Float64} 
    U_temp_sys::Matrix{Float64}   
    K_E1_sys::Matrix{Float64}     
    K_E2_sys::Matrix{Float64}     
    S_U_stage1_sys::Matrix{Float64} 
    S_U_stage2_sys::Matrix{Float64} 
    gamma_ars::Float64 

    function ARS2IMEX(
            gradientInterpolator::G1, 
            fallbackInterpolator::G2,
            mood::M,
            implicit_solver::IS, 
            source::AbstractSourceTerm,
            N_particles::Int, 
            N_components::Int;
            gamma_val::Float64 = 1.0 - 1.0 / sqrt(2.0)
        ) where {G1 <: Interpolations.GradientInterpolator, G2 <: Union{Interpolations.GradientInterpolator, Nothing}, M <: MOODCriterion, IS <: ImplicitSolvers.AbstractImplicitSolver} # Added Scopes
        
        
        new{G1,G2,M,IS}(
            gradientInterpolator, fallbackInterpolator, mood, implicit_solver, source,
            zeros(N_particles, N_components), zeros(N_particles, N_components),
            zeros(N_particles, N_components), zeros(N_particles, N_components),
            zeros(N_particles, N_components), zeros(N_particles, N_components),
            zeros(N_particles, N_components), zeros(N_particles, N_components),
            gamma_val
        )
    end
end

function initTimeStepper(
    ars2::ARS2IMEX,
    system_pg::Vector{<:ParticleGrids.ParticleGrid}, # Added ParticleGrids.
    settings::SimSettings.SimSetting # Added SimSettings.
)
    if isempty(system_pg) return end
    for k_comp in 1:length(system_pg)
        scalar_grid_k = system_pg[k_comp]
        initTimeStep(ars2.gradientInterpolator, scalar_grid_k, settings.interpAlpha, settings.interpRange)
        if !isnothing(ars2.fallbackInterpolator)
            initTimeStep(ars2.fallbackInterpolator, scalar_grid_k, settings.interpAlpha, settings.interpRange)
        end
    end
end

function (ars2::ARS2IMEX)(
        scalar_equations::Vector{<:ScalarHyperbolicEquations.ScalarHyperbolicEquation}, # Added ScalarHyperbolicEquations.
        system_pg::Vector{<:ParticleGrids.ParticleGrid}, # Added ParticleGrids.                       
        settings::SimSetting, # Added SimSettings.
        time_n::Real, 
        dt::Real
    )

    N_particles = length(system_pg[1].grid) 
    N_components = length(scalar_equations)
    gamma = ars2.gamma_ars

    if size(ars2.U_n_sys) != (N_particles, N_components)
        error("ARS2IMEX buffers not sized correctly. Expected ($(N_particles)x$(N_components)), got $(size(ars2.U_n_sys)). Re-initialize ARS2IMEX instance if N changes.")
    end

    for k_comp in 1:N_components
        for p_idx in interior_indices
            ars2.U_n_sys[p_idx, k_comp] = system_pg[k_comp].grid[p_idx].rho
        end
    end

    time_s1_implicit_eval = time_n + dt * gamma
    ars2.U_stage1_sys .= ars2.U_n_sys 
    
    u_particle_iter_buffer = Vector{Float64}(undef, N_components)
    rhs_const_buffer       = Vector{Float64}(undef, N_components)

    for p_idx in interior_indices
        particle_pos = system_pg[1].grid[p_idx].pos 
        u_particle_iter_buffer .= @view ars2.U_n_sys[p_idx, :] 
        rhs_const_buffer       .= @view ars2.U_n_sys[p_idx, :] 
        
        solve!(ars2.implicit_solver, # Added ImplicitSolvers.
            u_particle_iter_buffer, rhs_const_buffer, dt * gamma,
            ars2.source_term, particle_pos, time_s1_implicit_eval, N_components
        )
        ars2.U_stage1_sys[p_idx, :] .= u_particle_iter_buffer
    end

    for p_idx in interior_indices
        particle_pos = system_pg[1].grid[p_idx].pos
        u_stage1_p_view = @view ars2.U_stage1_sys[p_idx, :]
        s_u_stage1_p_view = @view ars2.S_U_stage1_sys[p_idx, :]
        ars2.source_term(s_u_stage1_p_view, u_stage1_p_view, particle_pos, time_s1_implicit_eval)
    end
    
    compute_explicit_tendency_with_mood!(
        ars2.K_E1_sys, ars2.U_n_sys, 
        ars2.gradientInterpolator, ars2.fallbackInterpolator, ars2.mood,
        scalar_equations, system_pg, settings, dt, true 
    )

    for p_idx in interior_indices
        for k_comp in 1:N_components
            ars2.U_temp_sys[p_idx, k_comp] = ars2.U_n_sys[p_idx, k_comp] + 
                                            dt * (1.0 - 2.0*gamma) * ars2.K_E1_sys[p_idx, k_comp] +
                                            dt * gamma * ars2.S_U_stage1_sys[p_idx, k_comp]
        end
    end

    time_s2_implicit_eval = time_n + dt 
    ars2.U_stage2_sys .= ars2.U_temp_sys

    for p_idx in interior_indices
        particle_pos = system_pg[1].grid[p_idx].pos
        u_particle_iter_buffer .= @view ars2.U_temp_sys[p_idx, :]
        rhs_const_s2_view = @view ars2.U_temp_sys[p_idx, :]
        
        ImplicitSolvers.solve!(ars2.implicit_solver, # Added ImplicitSolvers.
            u_particle_iter_buffer, rhs_const_s2_view, dt * gamma,
            ars2.source_term, particle_pos, time_s2_implicit_eval, N_components
        )
        ars2.U_stage2_sys[p_idx, :] .= u_particle_iter_buffer
    end

    for p_idx in interior_indices
        particle_pos = system_pg[1].grid[p_idx].pos
        u_stage2_p_view = @view ars2.U_stage2_sys[p_idx, :]
        s_u_stage2_p_view = @view ars2.S_U_stage2_sys[p_idx, :]
        ars2.source_term(s_u_stage2_p_view, u_stage2_p_view, particle_pos, time_s2_implicit_eval)
    end

    compute_explicit_tendency_with_mood!(
        ars2.K_E2_sys, ars2.U_stage1_sys, 
        ars2.gradientInterpolator, ars2.fallbackInterpolator, ars2.mood,
        scalar_equations, system_pg, settings, dt, false
    )

    for p_idx in interior_indices
        for k_comp in 1:N_components
            u_np1_k_p = ars2.U_n_sys[p_idx, k_comp] + 
                        0.5 * dt * (ars2.K_E1_sys[p_idx, k_comp] + ars2.K_E2_sys[p_idx, k_comp]) +
                        0.5 * dt * (ars2.S_U_stage1_sys[p_idx, k_comp] + ars2.S_U_stage2_sys[p_idx, k_comp])
            system_pg[k_comp].grid[p_idx].rho = u_np1_k_p
        end
    end
    
    for k_comp in 1:N_components
        for p_obj in system_pg[k_comp].grid
            p_obj.moodEvent = false 
        end
    end
end




# --- GeneralIMEXTimeStepper Struct ---
struct GeneralIMEXTimeStepper{
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver, # Assuming AbstractImplicitSolver is defined
    ST_OBJ <: SourceTerms.AbstractSourceTerm,    # Assuming AbstractSourceTerm is defined
    BT <: IMEXButcherTableau
} <: TimeStepper # Qualify TimeStepper if in a different module

    # User's modular components
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::M
    implicit_solver::IS
    source_term_object::ST_OBJ
    
    # Butcher Tableau for the specific IMEX scheme
    butcher_tableau::BT
    
    # Buffers (N_particles x N_components)
    U_n_sys::Matrix{Float64}
    Y_stages_sys::Vector{Matrix{Float64}}   # Stores Y_i (solution at each stage)
    K_E_stages_sys::Vector{Matrix{Float64}} # Stores F_E(Y_i) for each stage
    K_I_stages_sys::Vector{Matrix{Float64}} # Stores F_I(Y_i) for each stage
    
    num_stages::Int

    function GeneralIMEXTimeStepper(
            gradientInterpolator::G1,
            fallbackInterpolator::G2,
            mood::M,
            implicit_solver::IS,
            source_term_object::ST_OBJ,
            butcher_tableau::BT,
            N_particles::Int,
            N_components::Int
        ) where {
            G1 <: Interpolations.GradientInterpolator, G2 <: Union{Interpolations.GradientInterpolator, Nothing},
            M <: MOODCriterion, IS <: ImplicitSolvers.AbstractImplicitSolver,
            ST_OBJ <: SourceTerms.AbstractSourceTerm, BT <: IMEXButcherTableau
        }
        
        s = size(butcher_tableau.A, 1) # Number of stages

        Y_stages = [zeros(N_particles, N_components) for _ in 1:s]
        K_E_stages = [zeros(N_particles, N_components) for _ in 1:s]
        K_I_stages = [zeros(N_particles, N_components) for _ in 1:s]
        U_n_buffer = zeros(N_particles, N_components)

        new{G1,G2,M,IS,ST_OBJ,BT}(
            gradientInterpolator, fallbackInterpolator, mood, implicit_solver, source_term_object, butcher_tableau,
            U_n_buffer, Y_stages, K_E_stages, K_I_stages, s
        )
    end
end

# initTimeStepper for GeneralIMEXTimeStepper (for geometric precomputations of interpolators)
function initTimeStepper(
    imex_ts::GeneralIMEXTimeStepper,
    system_pg::Vector{<:ParticleGrids.ParticleGrid}, # Vector of ParticleGrid (1D or 2D)
    settings::SimSettings.SimSetting
)
    if isempty(system_pg) return end
    # For each component's grid, call the standard initTimeStep (without fVec)
    # for the interpolators. This is for purely geometric setup.
    # Solution-dependent init (like for MUSCLlimited's limited_slopes_cache)
    # is handled within compute_explicit_tendency_with_mood! before each F_E eval.
    for k_comp in 1:length(system_pg)
        scalar_grid_k = system_pg[k_comp]
        initTimeStep(imex_ts.gradientInterpolator, scalar_grid_k, settings.interpAlpha, settings.interpRange)
        if !isnothing(imex_ts.fallbackInterpolator)
            initTimeStep(imex_ts.fallbackInterpolator, scalar_grid_k, settings.interpAlpha, settings.interpRange)
        end
    end
end

# --- Corrected Functor for GeneralIMEXTimeStepper ---
function (imex_ts::GeneralIMEXTimeStepper)(
        scalar_equations::Vector{<:ScalarHyperbolicEquations.ScalarHyperbolicEquation},
        system_pg::Vector{<:ParticleGrids.ParticleGrid},
        settings::SimSettings.SimSetting,
        time_n::Real,
        dt::Real
    )

    interior_indices = system_pg[1].interior_indices
    N_total_particles = length(system_pg[1].grid)
    N_components = length(scalar_equations)
    s = imex_ts.num_stages
    bt = imex_ts.butcher_tableau

    if size(imex_ts.U_n_sys,1) != N_total_particles || size(imex_ts.U_n_sys,2) != N_components
        error("GeneralIMEXTimeStepper buffers not sized correctly. Re-initialize instance.")
    end

    # --- 0. Store U^n from system_pg ---
    # CORRECTED: This loop MUST iterate over ALL particles (1:N_total_particles)
    # to correctly cache the state of interior AND ghost cells from the physical grid.
    for k_comp in 1:N_components
        for p_idx in 1:N_total_particles
            imex_ts.U_n_sys[p_idx, k_comp] = system_pg[k_comp].grid[p_idx].rho
        end
    end

    # Temporary particle-local vectors for implicit solve
    u_particle_iter_buffer = Vector{Float64}(undef, N_components)
    Y_i_base_particle      = Vector{Float64}(undef, N_components)


    # --- Loop through stages i = 1 to s ---
    for i in 1:s
        current_Y_i_sys = imex_ts.Y_stages_sys[i] # Alias to the cache for Y_i
        
        # Initialize Y_i_base = U^n for this stage's calculation.
        # This copies the FULL state, including correct ghost cells from U_n_sys.
        current_Y_i_sys .= imex_ts.U_n_sys

        # --- Calculate Y_i_base for INTERIOR points ---
        # Add contributions from previous stages only to the interior points
        # Explicit sum part
        for j in 1:(i-1)
            if bt.At[i,j] != 0.0
                for p_idx_loop in interior_indices, k_comp_loop in 1:N_components
                    current_Y_i_sys[p_idx_loop, k_comp_loop] += dt * bt.At[i,j] * imex_ts.K_E_stages_sys[j][p_idx_loop, k_comp_loop]
                end
            end
        end
        # Implicit sum from previous stages
        for j in 1:(i-1)
            if bt.A[i,j] != 0.0
                for p_idx_loop in interior_indices, k_comp_loop in 1:N_components
                    current_Y_i_sys[p_idx_loop, k_comp_loop] += dt * bt.A[i,j] * imex_ts.K_I_stages_sys[j][p_idx_loop, k_comp_loop]
                end
            end
        end
        
        # --- Implicit Solve for stage Y_i for INTERIOR points ---
        if abs(bt.A[i,i]) > 1e-14
            time_implicit_eval = time_n + bt.c[i] * dt
            for p_idx in interior_indices
                particle_pos = system_pg[1].grid[p_idx].pos
                u_particle_iter_buffer .= @view current_Y_i_sys[p_idx, :] 
                Y_i_base_particle      .= @view current_Y_i_sys[p_idx, :] 
                                          
                ImplicitSolvers.solve!(imex_ts.implicit_solver,
                    u_particle_iter_buffer, Y_i_base_particle, dt * bt.A[i,i],
                    imex_ts.source_term_object, particle_pos, time_implicit_eval, N_components
                )
                current_Y_i_sys[p_idx, :] .= u_particle_iter_buffer
            end
        end
        
        # --- Ghost Cell Update for Intermediate Stage Y_i ---
        # Before computing K_E(Y_i), we need Y_i to have correct ghost values.
        for k_comp in 1:N_components
            # Temporarily put the computed interior Y_i state into its particle grid
            first_index = first(interior_indices)
            system_pg[k_comp].grid[first_index].rho = current_Y_i_sys[first_index, k_comp]
            last_index = last(interior_indices)
            system_pg[k_comp].grid[last_index].rho = current_Y_i_sys[last_index, k_comp]
            # Apply boundary conditions, which will update the ghost cell .rho values
            apply_boundary_conditions!(system_pg[k_comp])
            # Copy the updated ghost cell values back to our cache matrix
            for p_idx in 1:N_total_particles
                if !(p_idx in interior_indices)
                    current_Y_i_sys[p_idx, k_comp] = system_pg[k_comp].grid[p_idx].rho
                end
            end
        end
        # Now current_Y_i_sys (imex_ts.Y_stages_sys[i]) is fully correct for this stage.

        # --- Evaluate and store tendencies K_Ei and K_Ii using the full Y_i state ---
        time_explicit_eval = time_n + bt.ct[i] * dt
        compute_explicit_tendency_with_mood!(
            imex_ts.K_E_stages_sys[i], current_Y_i_sys, 
            imex_ts.gradientInterpolator, imex_ts.fallbackInterpolator, imex_ts.mood,
            scalar_equations, system_pg, settings, dt, (i==1), interior_indices
        )

        time_implicit_eval_for_KI = time_n + bt.c[i] * dt 
        for p_idx in 1:N_total_particles # Evaluate source over all points for the sum
            particle_pos = system_pg[1].grid[p_idx].pos
            Y_i_p_view = @view current_Y_i_sys[p_idx, :]
            K_Ii_p_view = @view imex_ts.K_I_stages_sys[i][p_idx, :]
            imex_ts.source_term_object(K_Ii_p_view, Y_i_p_view, particle_pos, time_implicit_eval_for_KI)
        end
    end # End of stages loop

    # --- Final Update ---
    U_np1_sys_temp = copy(imex_ts.U_n_sys) 
    for i in 1:s
        if abs(bt.bt[i]) > 1e-14 || abs(bt.b[i]) > 1e-14 # Check both weights
            for p_idx_loop in interior_indices, k_comp_loop in 1:N_components
                U_np1_sys_temp[p_idx_loop, k_comp_loop] += 
                    dt * (bt.bt[i] * imex_ts.K_E_stages_sys[i][p_idx_loop, k_comp_loop] + 
                          bt.b[i] * imex_ts.K_I_stages_sys[i][p_idx_loop, k_comp_loop])
            end
        end
    end

    # Update physical particleGrid (only interior points)
    for k_comp in 1:N_components
        for p_idx in interior_indices
            system_pg[k_comp].grid[p_idx].rho = U_np1_sys_temp[p_idx, k_comp]
        end
    end
    
    # Reset moodEvent flags on all particles for the next step
    for k_comp in 1:N_components
        for p_obj in system_pg[k_comp].grid
            p_obj.moodEvent = false 
        end
    end
end

# # --- REVISED Functor for GeneralIMEXTimeStepper (Explicit-First Logic) ---
# function (imex_ts::GeneralIMEXTimeStepper)(
#         scalar_equations::Vector{<:ScalarHyperbolicEquations.ScalarHyperbolicEquation},
#         system_pg::Vector{<:ParticleGrids.ParticleGrid},
#         settings::SimSettings.SimSetting,
#         time_n::Real,
#         dt::Real
#     )

#     interior_indices = system_pg[1].interior_indices
#     N_total_particles = length(system_pg[1].grid)
#     N_components = length(scalar_equations)
#     s = imex_ts.num_stages
#     bt = imex_ts.butcher_tableau

#     if size(imex_ts.U_n_sys,1) != N_total_particles || size(imex_ts.U_n_sys,2) != N_components
#         error("GeneralIMEXTimeStepper buffers not sized correctly. Re-initialize instance.")
#     end

#     # --- 0. Store U^n from system_pg (including ghost cells) ---
#     for k_comp in 1:N_components
#         for p_idx in 1:N_total_particles
#             imex_ts.U_n_sys[p_idx, k_comp] = system_pg[k_comp].grid[p_idx].rho
#         end
#     end

#     # Temporary particle-local vectors for implicit solve
#     u_particle_iter_buffer = Vector{Float64}(undef, N_components)
#     Y_i_base_for_implicit_solve = Vector{Float64}(undef, N_components)

#     # --- Loop through stages i = 1 to s ---
#     for i in 1:s
#         # This buffer will hold the state used to compute the explicit tendency K_E
#         Y_i_for_explicit_eval = imex_ts.Y_stages_sys[i] # Reuse stage buffer temporarily
#         Y_i_for_explicit_eval .= imex_ts.U_n_sys

#         # --- Step 1: Build the state for the EXPLICIT tendency evaluation ---
#         # Y_i_E = U^n + dt * sum_{j=1}^{i-1} (At[i,j]*K_Ej + A[i,j]*K_Ij)
#         # Note: Some IMEX schemes use the same stage value for both explicit and implicit
#         # tendencies. The ARS schemes do this. We build the full base state first.
#         for j in 1:(i-1)
#             if bt.At[i,j] != 0.0
#                 @. Y_i_for_explicit_eval[interior_indices, :] += dt * bt.At[i,j] * imex_ts.K_E_stages_sys[j][interior_indices, :]
#             end
#             if bt.A[i,j] != 0.0
#                 @. Y_i_for_explicit_eval[interior_indices, :] += dt * bt.A[i,j] * imex_ts.K_I_stages_sys[j][interior_indices, :]
#             end
#         end
        
#         # --- Step 2: Update ghost cells for this intermediate state ---
#         # The explicit operator needs correct ghost values for its stencil.
#         for k_comp in 1:N_components
#             for p_idx in interior_indices; system_pg[k_comp].grid[p_idx].rho = Y_i_for_explicit_eval[p_idx, k_comp]; end
#             apply_boundary_conditions!(system_pg[k_comp])
#             for p_idx in 1:N_total_particles; if !(p_idx in interior_indices); Y_i_for_explicit_eval[p_idx, k_comp] = system_pg[k_comp].grid[p_idx].rho; end; end
#         end

#         # --- Step 3: Evaluate and store the EXPLICIT tendency K_Ei FIRST ---
#         time_explicit_eval = time_n + bt.ct[i] * dt
#         compute_explicit_tendency_with_mood!(
#             imex_ts.K_E_stages_sys[i], Y_i_for_explicit_eval, 
#             imex_ts.gradientInterpolator, imex_ts.fallbackInterpolator, imex_ts.mood,
#             scalar_equations, system_pg, settings, dt, (i==1), interior_indices
#         )

#         # --- Step 4: Build the RHS for the IMPLICIT solve ---
#         # This now includes the explicit tendency from the current stage.
#         # Y_i_base = U^n + dt*sum_{j=1}^{i} At[i,j]*K_Ej + dt*sum_{j=1}^{i-1} A[i,j]*K_Ij
#         # We can just add the new K_E term to our existing Y_i_for_explicit_eval
#         if bt.At[i,i] != 0.0 # This is usually zero for ARS schemes but included for generality
#             @. Y_i_for_explicit_eval[interior_indices, :] += dt * bt.At[i,i] * imex_ts.K_E_stages_sys[i][interior_indices, :]
#         end
        
#         # --- Step 5: Implicit Solve for the final stage value Y_i ---
#         current_Y_i_sys = Y_i_for_explicit_eval # Y_i_for_explicit_eval now holds the full RHS
#         if abs(bt.A[i,i]) > 1e-14
#             time_implicit_eval = time_n + bt.c[i] * dt
#             for p_idx in interior_indices
#                 particle_pos = system_pg[1].grid[p_idx].pos
#                 u_particle_iter_buffer .= @view current_Y_i_sys[p_idx, :] 
#                 Y_i_base_for_implicit_solve .= @view current_Y_i_sys[p_idx, :] 
                                          
#                 ImplicitSolvers.solve!(imex_ts.implicit_solver,
#                     u_particle_iter_buffer, Y_i_base_for_implicit_solve, dt * bt.A[i,i],
#                     imex_ts.source_term_object, particle_pos, time_implicit_eval, N_components
#                 )
#                 current_Y_i_sys[p_idx, :] .= u_particle_iter_buffer
#             end
#         end
        
#         # --- Step 6: Evaluate and store the IMPLICIT tendency K_Ii ---
#         time_implicit_eval_for_KI = time_n + bt.c[i] * dt 
#         for p_idx in 1:N_total_particles
#             particle_pos = system_pg[1].grid[p_idx].pos
#             Y_i_p_view = @view current_Y_i_sys[p_idx, :]
#             K_Ii_p_view = @view imex_ts.K_I_stages_sys[i][p_idx, :]
#             imex_ts.source_term_object(K_Ii_p_view, Y_i_p_view, particle_pos, time_implicit_eval_for_KI)
#         end
#     end # End of stages loop

#     # --- Final Update (this part remains the same) ---
#     U_np1_sys_temp = copy(imex_ts.U_n_sys) 
#     for i in 1:s
#         if abs(bt.bt[i]) > 1e-14 || abs(bt.b[i]) > 1e-14
#             @. U_np1_sys_temp[interior_indices, :] += dt * (bt.bt[i] * imex_ts.K_E_stages_sys[i][interior_indices, :] + bt.b[i] * imex_ts.K_I_stages_sys[i][interior_indices, :])
#         end
#     end

#     # Update physical particleGrid
#     for k_comp in 1:N_components
#         for p_idx in interior_indices
#             system_pg[k_comp].grid[p_idx].rho = U_np1_sys_temp[p_idx, k_comp]
#         end
#     end
    
#     for k_comp in 1:N_components
#         for p_obj in system_pg[k_comp].grid
#             p_obj.moodEvent = false 
#         end
#     end
# end


"""
    ARS233(
        gradientInterpolator, 
        fallbackInterpolator, # Union{GradientInterpolator, Nothing}
        mood_criterion,
        implicit_solver, 
        source_term_object,
        N_particles::Int, 
        N_components::Int;
        gamma_coefficient::Float64 = (3.0 + sqrt(3.0))/6.0 # Specific to ARS233
    )

Constructs a `GeneralIMEXTimeStepper` pre-configured with the IMEXARS233 Butcher tableau.
This scheme is a 3-stage, 3rd order Additive Runge-Kutta scheme.
"""
function ARS233( # Changed name to avoid conflict with potential struct name if desired
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
    N_particles::Int,
    N_components::Int;
    gamma_coefficient::Float64 = (3.0 + sqrt(3.0))/6.0 # Allow custom gamma for this specific scheme
) where {
    G1 <: Interpolations.GradientInterpolator, # Example: Qualify with your module name
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    # Get the specific Butcher tableau for IMEXARS233
    # Pass the gamma if your tableau function accepts it.
    tableau = IMEXARS233ButcherTableau(gamma_coefficient) 

    return GeneralIMEXTimeStepper( # Assuming GeneralIMEXTimeStepper is defined in current scope
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific Butcher tableau
        N_particles,
        N_components
    )
end
# In a file like IMEXTableaus.jl or alongside GeneralIMEXTimeStepper definition
# Ensure your IMEXButcherTableau struct is defined as you provided previously.
# struct IMEXButcherTableau{M <: AbstractArray{Float64, 2}, V <: AbstractArray{Float64, 1}} ... end

"""
    PR_IMEX_SSP3_ButcherTableau()::IMEXButcherTableau

Returns the Butcher tableau for the Pareschi & Russo (2005) IMEX-SSP3(3,3,3)
scheme (Scheme 4.2 in their JCP paper "Implicit-explicit Runge-Kutta schemes
and applications to hyperbolic systems with relaxation").
This is a 3-stage, 3rd order, L-stable scheme.
"""


# In your MeshfreeTimeSteppers.jl or SystemIMEXTimeSteppers.jl
# Ensure GeneralIMEXTimeStepper and all necessary types (GradientInterpolator, etc.)
# and PR_IMEX_SSP3_ButcherTableau are accessible.

"""
    PareschiRussoIMEXSSP3(
        gradientInterpolator, fallbackInterpolator, mood_criterion,
        implicit_solver, source_term_object,
        N_particles::Int, N_components::Int
    )

Constructs a GeneralIMEXTimeStepper pre-configured with the Pareschi & Russo (2005)
IMEX-SSP3(3,3,3) Butcher tableau (Scheme 4.2).
This is a 3-stage, 3rd order, L-stable scheme.
"""
function PareschiRussoIMEXSSP3(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
    N_particles::Int,
    N_components::Int
) where {
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    tableau = PR_IMEX_SSP3_ButcherTableau() 

    return GeneralIMEXTimeStepper(
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific Butcher tableau
        N_particles,
        N_components
    )
end

"""
    ARS222_IMEX(
        gradientInterpolator, fallbackInterpolator, mood_criterion,
        implicit_solver, source_term_object,
        N_particles::Int, N_components::Int;
        gamma_coefficient::Union{Float64,Nothing}=nothing 
    )

Constructs a GeneralIMEXTimeStepper pre-configured with the standard
ARS(2,2,2) IMEX Butcher tableau by Ascher, Ruuth, Spiteri (1997).
This is a 2-stage, 2nd order, L-stable scheme.
"""
function ARS222(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
    N_particles::Int,
    N_components::Int;
    gamma_coefficient::Union{Float64,Nothing}=nothing # Allows override of default gamma
) where {
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    tableau = ARS222_ButcherTableau(gamma_coefficient) 

    return GeneralIMEXTimeStepper(
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific ARS(2,2,2) Butcher tableau
        N_particles,
        N_components
    )
end


function SSP2332(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
    N_particles::Int,
    N_components::Int
) where {
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    tableau = SSP2332ButcherTableau() 

    return GeneralIMEXTimeStepper(
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific ARS(2,2,2) Butcher tableau
        N_particles,
        N_components
    )
end

function RalstonRK2(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
    N_particles::Int,
    N_components::Int
) where {
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    tableau = RalstonRK2ButcherTableau() 

    return GeneralIMEXTimeStepper(
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific ARS(2,2,2) Butcher tableau
        N_particles,
        N_components
    )
end