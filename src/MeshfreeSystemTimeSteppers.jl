
# Default case: Everything is decoupled
export GeneralIMEXTimeStepper, ARS233, PareschiRussoIMEXSSP3, ARS222, SSP2332, SimpleSplitting, RalstonRK2, ARS232

include("ButcherTableaus.jl")

# --- In your TimeIntegration.jl file ---

using InteractiveUtils

function run_functor_test(rs, S_out, U_kin, pos, time)
    # This is the call we want to inspect
    rs(S_out, U_kin, pos, time)
end

# In your main code, after `my_source_term` is defined:
# 

#==============================================================================
  Simple Splitting Timestepper (Optimized for SoA Grids)
==============================================================================#

# --- Generic System Stepper Fallbacks (as provided by you) ---
function (ts::TimeStepper)(eqs::DiagonalHyperbolicSystem{N,D}, particleGrids::ParticleGridSystem{N}, settings::SimSetting, time::Real, dt::Real) where {N,D}
    @warn "No dedicated system timestepper found. System will be treated independently using the given scalar timestepper."
    for (i,particleGrid) = enumerate(particleGrids)
        ts(eqs[i], particleGrid, settings, time, dt)
    end
end

function (ts::TimeStepper)(eq::ScalarHyperbolicPDE, particleGrids::ParticleGridSystem{N}, settings::SimSetting, time::Real, dt::Real) where {N}
    @warn "Only one scalar hyperbolic equation for a system is found. The scalar equation will be used for all components!"
    ts(ntuple(i -> eq, N), particleGrids, settings, time, dt)
end


# --- REFACTORED SimpleSplitting Struct and Constructor ---
mutable struct SimpleSplitting{T <: TimeStepper, S <: AbstractSourceTerm} <: MeshfreeSystemTimeStepper
    timestepper::T 
    source_term::S
    macro_state_buffer::Matrix{Float64} 

    function SimpleSplitting(
        timestepper::T, 
        source_term::S
    ) where {T <: TimeStepper, S <: AbstractSourceTerm}
        @assert !isa(timestepper, MeshfreeSystemTimeStepper) "A scalar timestepper must be provided for the advection step."
        @assert isa(source_term, RelaxationSourceTerm) "Source term must be a RelaxationSourceTerm."
        
        # Initialize buffer as an empty matrix. It will be resized on the first time step.
        macro_buffer = Matrix{Float64}(undef, 0, 0)
        
        new{T, S}(timestepper, source_term, macro_buffer)
    end
end

function initTimeStepper(method::SimpleSplitting, particleGrids::ParticleGridSystem, settings::SimSetting)
    # Initialize the underlying scalar timestepper for each component grid
    for pg in particleGrids
        initTimeStepper(method.timestepper, pg, settings)
    end
end

"""
Functor for the SimpleSplitting timestepper, optimized for SoA grids.
"""
function (ss::SimpleSplitting)(
    eqs::DiagonalHyperbolicSystem{N,D},
    particleGrids::ParticleGridSystem{N}, 
    settings::SimSetting, 
    time::Real, 
    dt::Real
) where {N,D}
    
    # --- 1. Advection Step ---
    # Apply the scalar timestepper to each kinetic component.
    # This updates the `rhos` array in each grid to the post-advection state v_k^*.
    for k_comp in 1:N
        ss.timestepper(eqs[k_comp], particleGrids[k_comp], settings, time, dt)
    end

    # --- 2. Recombination Step ---
    N_total_particles = particleGrids[1].N
    rs = ss.source_term

    # --- NEW: Ensure the buffer is correctly sized ---
    if size(ss.macro_state_buffer, 1) != N_total_particles || size(ss.macro_state_buffer, 2) != rs.num_macro_variables
        ss.macro_state_buffer = Matrix{Float64}(undef, N_total_particles, rs.num_macro_variables)
    end

    # Reconstruct the macroscopic state U_macro^* at ALL particle locations
    for p_idx in 1:N_total_particles
        for i_macro in 1:rs.num_macro_variables
            # Sum the advected kinetic variables (v_k^*) using direct array access
            indices = rs.kinetic_indices[i_macro]
            ss.macro_state_buffer[p_idx, i_macro] = sum(particleGrids[k_idx].rhos[p_idx] for k_idx in indices)
        end
    end

    # --- 3. Relaxation Step ---
    coeff_ep = rs.epsilon / (rs.epsilon + dt)
    coeff_dt = dt / (rs.epsilon + dt)

    for k_comp in 1:N
        pg_k = particleGrids[k_comp]
        maxwellian_func_k = rs.maxwellians[k_comp]

        for p_idx in pg_k.interior_indices
            # v_k^*(p_idx) from the advection step
            v_k_star_at_p = pg_k.rhos[p_idx]
            
            # Get the macroscopic state vector (U_macro_i^*) using ntuple for performance
            U_macro_star_at_p = ntuple(i -> ss.macro_state_buffer[p_idx, i], rs.num_macro_variables)

            # Evaluate the Maxwellian (it expects a tuple)
            equilibrium_val_k = maxwellian_func_k(U_macro_star_at_p)
            
            # Update the rhos array to the final state v_k^{n+1}
            pg_k.rhos[p_idx] = coeff_ep * v_k_star_at_p + coeff_dt * equilibrium_val_k
        end
    end
end
# --- In your TimeIntegration.jl file ---
# --- In your TimeIntegration.jl file ---

#==============================================================================
  General IMEX Timestepper (Optimized for SoA Grids and Adaptive Workspaces)
==============================================================================#

# --- REFACTORED GeneralIMEXTimeStepper Struct and Constructor ---
mutable struct GeneralIMEXTimeStepper{G1, G2, M, IS, ST_OBJ, BT} <: MeshfreeSystemTimeStepper
    # User's modular components
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::M
    implicit_solver::IS
    source_term_object::ST_OBJ
    butcher_tableau::BT
    
    # --- Reusable Buffers (Workspace) ---
    U_n_sys::Matrix{Float64}
    Y_stages_sys::Vector{Matrix{Float64}}
    K_E_stages_sys::Vector{Matrix{Float64}}
    K_I_stages_sys::Vector{Matrix{Float64}}
    rho_buffer::Vector{Float64}
    
    # --- Particle-local buffers for implicit solve ---
    u_particle_iter_buffer::Vector{Float64}
    Y_i_base_particle_buffer::Vector{Float64}
    
    num_stages::Int

    function GeneralIMEXTimeStepper(
            gradientInterpolator::G1, fallbackInterpolator::G2, mood::M,
            implicit_solver::IS, source_term_object::ST_OBJ, butcher_tableau::BT
        ) where {G1, G2, M, IS, ST_OBJ, BT}
        
        s = size(butcher_tableau.A, 1) # Number of stages

        # Initialize with empty buffers; they will be resized on the first call
        new{G1, G2, M, IS, ST_OBJ, BT}(
            gradientInterpolator, fallbackInterpolator, mood, implicit_solver, 
            source_term_object, butcher_tableau,
            Matrix{Float64}(undef,0,0), [Matrix{Float64}(undef,0,0) for _ in 1:s],
            [Matrix{Float64}(undef,0,0) for _ in 1:s], [Matrix{Float64}(undef,0,0) for _ in 1:s], Vector{Float64}(undef, 0),
            Float64[], Float64[], s
        )
    end
end

# --- REFACTORED initTimeStepper (no changes needed, it's already generic) ---
function initTimeStepper(imex_ts::GeneralIMEXTimeStepper, system_pg::ParticleGridSystem, settings::SimSetting)
    for pg in system_pg
        initTimeStep(imex_ts.gradientInterpolator, pg, settings.interpAlpha, settings.interpRange)
        if !isnothing(imex_ts.fallbackInterpolator)
            initTimeStep(imex_ts.fallbackInterpolator, pg, settings.interpAlpha, settings.interpRange)
        end
    end
end


# --- REFACTORED Functor for GeneralIMEXTimeStepper ---
function (imex_ts::GeneralIMEXTimeStepper{G1, G2, M, IS, ST_OBJ, BT})(
        scalar_equations::DiagonalHyperbolicSystem{N,D},
        system_pg::ParticleGridSystem{N},
        settings::SimSetting,
        time_n::Real,
        dt::Real
    ) where {G1, G2, M, IS, ST_OBJ, BT, N, D}

    interior = system_pg[1].interior_indices
    N_particles = system_pg[1].N
    N_components = N
    s = imex_ts.num_stages
    bt = imex_ts.butcher_tableau

    # --- Ensure buffers are correctly sized for the current grid ---
    if size(imex_ts.U_n_sys, 1) != N_particles
        # Re-create matrices (necessary) using undef for a small speedup
        imex_ts.U_n_sys = Matrix{Float64}(undef, N_particles, N_components)
        # Re-create the stage matrices
        for i in 1:s
            imex_ts.Y_stages_sys[i] = Matrix{Float64}(undef, N_particles, N_components)
            imex_ts.K_E_stages_sys[i] = Matrix{Float64}(undef, N_particles, N_components)
            imex_ts.K_I_stages_sys[i] = Matrix{Float64}(undef, N_particles, N_components)
        end
        resize!(imex_ts.rho_buffer, N_particles)
        resize!(imex_ts.u_particle_iter_buffer, N_components)
        resize!(imex_ts.Y_i_base_particle_buffer, N_components)
    end

    # --- 0. Store U^n from system_pg ---
    for k in 1:N; imex_ts.U_n_sys[:, k] = system_pg[k].rhos; end

    # --- Loop through stages i = 1 to s ---
    for i in 1:s
        current_Y_i_sys = imex_ts.Y_stages_sys[i]
        current_Y_i_sys .= imex_ts.U_n_sys # Start with U^n

        # --- Calculate stage value Y_i for INTERIOR points ---
        for j in 1:(i-1)
            if bt.At[i,j] != 0.0; @. current_Y_i_sys[interior, :] += dt * bt.At[i,j] * imex_ts.K_E_stages_sys[j][interior, :]; end
            if bt.A[i,j] != 0.0;  @. current_Y_i_sys[interior, :] += dt * bt.A[i,j] * imex_ts.K_I_stages_sys[j][interior, :]; end
        end
        
        # --- Implicit Solve for stage Y_i for INTERIOR points ---
        if abs(bt.A[i,i]) > 1e-14
            time_implicit = time_n + bt.c[i] * dt
            for p_idx in interior
                imex_ts.u_particle_iter_buffer .= @view current_Y_i_sys[p_idx, :] 
                imex_ts.Y_i_base_particle_buffer .= @view current_Y_i_sys[p_idx, :] 
                                          
                ImplicitSolvers.solve!(imex_ts.implicit_solver,
                    imex_ts.u_particle_iter_buffer, imex_ts.Y_i_base_particle_buffer, dt * bt.A[i,i],
                    imex_ts.source_term_object, system_pg[1].positions[p_idx], time_implicit, N_components
                )
                current_Y_i_sys[p_idx, :] .= imex_ts.u_particle_iter_buffer
            end
        end
        
        # --- Ghost Cell Update for Intermediate Stage Y_i ---
        for k in 1:N
            system_pg[k].rhos[interior] .= @view current_Y_i_sys[interior, k]
            apply_boundary_conditions!(system_pg[k])
            current_Y_i_sys[:, k] .= system_pg[k].rhos # Copy back full state with ghosts
        end
        
        # --- Evaluate and store tendencies K_E and K_I ---
        compute_explicit_tendency_with_mood!(
            imex_ts.K_E_stages_sys[i], current_Y_i_sys, imex_ts.rho_buffer,
            imex_ts.gradientInterpolator, imex_ts.fallbackInterpolator, imex_ts.mood,
            scalar_equations, system_pg, settings, dt, (i==1)
        )

        time_implicit_for_KI = time_n + bt.c[i] * dt 
        
        for p_idx in 1:N_particles
            # @code_warntype run_functor_test(imex_ts.source_term_object,
            #     @view(imex_ts.K_I_stages_sys[i][p_idx, :]), 
            #     @view(current_Y_i_sys[p_idx, :]), 
            #     system_pg[1].positions[p_idx], 
            #     time_implicit_for_KI
            # )
            imex_ts.source_term_object(
                @view(imex_ts.K_I_stages_sys[i][p_idx, :]), 
                @view(current_Y_i_sys[p_idx, :]), 
                system_pg[1].positions[p_idx], 
                time_implicit_for_KI
            )
        end
    end # End of stages loop

    # --- Final Update ---
    U_np1 = imex_ts.U_n_sys # Reuse this buffer for the final result
    for i in 1:s
        if abs(bt.bt[i]) > 1e-14; @. U_np1[interior, :] += dt * bt.bt[i] * imex_ts.K_E_stages_sys[i][interior, :]; end
        if abs(bt.b[i]) > 1e-14;  @. U_np1[interior, :] += dt * bt.b[i] * imex_ts.K_I_stages_sys[i][interior, :]; end
    end

    # --- Update physical particle grids ---
    for k in 1:N
        system_pg[k].rhos[interior] .= @view U_np1[interior, k]
        system_pg[k].mood_events .= false
    end
end


# --- REFACTORED Helper to compute explicit tendency ---
function compute_explicit_tendency_with_mood!(
    K_E_out::Matrix{Float64},
    U_state::Matrix{Float64},
    rho_buffer::Vector{Float64},
    grad_interp, fallback_interp, mood,
    eqs::DiagonalHyperbolicSystem{N,D},
    grids::ParticleGridSystem{N},
    settings::SimSetting,
    dt::Real,
    is_first_stage::Bool
) where {N,D}
    
    interior = grids[1].interior_indices
    
    for k in 1:N
        grid_k = grids[k]
        eq_k = eqs[k]
        
        # Backup the current state of the physical grid component
        rho_buffer .= grid_k.rhos

#        try
        # Temporarily update the physical grid's state to U_state for this stage
        grid_k.rhos .= @view U_state[:, k]
        
        # Pre-computation steps for interpolators for this stage
        initTimeStep(grad_interp, grid_k, settings.interpAlpha, settings.interpRange)
        if !isnothing(fallback_interp)
            initTimeStep(fallback_interp, grid_k, settings.interpAlpha, settings.interpRange)
        end
        copyCurvatures!(grid_k)

        # Calculate divergence for each interior particle
        for p_idx in interior
            div_high = grad_interp(grid_k, p_idx, grid_k.rhos, eq_k, settings; setCurvature=true)
            rho_candidate = U_state[p_idx, k] - dt * div_high
            
            if !isnothing(fallback_interp) && mood(grid_k, p_idx, @view(U_state[:, k]), rho_candidate; firstStage=is_first_stage)
                div_fallback = fallback_interp(grid_k, p_idx, grid_k.rhos, eq_k, settings; setCurvature=false)
                K_E_out[p_idx, k] = -div_fallback
            else
                K_E_out[p_idx, k] = -div_high
            end
        end
#        finally
        # Always restore the original grid state
        grid_k.rhos .= rho_buffer
#        end
    end
end




# Version of IMEXTimestepper that separates the explicit and implicit stages
struct GeneralIMEXTimeStepperS{
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver, # Assuming AbstractImplicitSolver is defined
    ST_OBJ <: SourceTerms.AbstractSourceTerm,    # Assuming AbstractSourceTerm is defined
    BT <: IMEXButcherTableau
} <: MeshfreeSystemTimeStepper # Qualify TimeStepper if in a different module

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

    function GeneralIMEXTimeStepperS(
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
    imex_ts::GeneralIMEXTimeStepperS,
    system_pg::ParticleGridSystem{N}, # Vector of ParticleGrid (1D or 2D)
    settings::SimSettings.SimSetting
) where {N}
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
function (imex_ts::GeneralIMEXTimeStepperS)(
        scalar_equations::DiagonalHyperbolicSystem{N,D},
        system_pg::ParticleGridSystem{N},
        settings::SimSettings.SimSetting,
        time_n::Real,
        dt::Real
    ) where {N,D}

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

 
    end # End of stages loop

    for i in 1:s
        current_Y_i_sys = imex_ts.Y_stages_sys[i] # Alias to the cache for Y_i
        
        # Initialize Y_i_base = U^n for this stage's calculation.
        # This copies the FULL state, including correct ghost cells from U_n_sys.
        current_Y_i_sys .= imex_ts.U_n_sys
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
       time_implicit_eval_for_KI = time_n + bt.c[i] * dt 
        for p_idx in 1:N_total_particles # Evaluate source over all points for the sum
            particle_pos = system_pg[1].grid[p_idx].pos
            Y_i_p_view = @view current_Y_i_sys[p_idx, :]
            K_Ii_p_view = @view imex_ts.K_I_stages_sys[i][p_idx, :]
            imex_ts.source_term_object(K_Ii_p_view, Y_i_p_view, particle_pos, time_implicit_eval_for_KI)
        end
    end

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
#         scalar_equations::Vector{<:ScalarHyperbolicPDEs.ScalarHyperbolicPDE},
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
    )
end


function SSP2332(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
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
    )
end

function RalstonRK2(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
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
    )
end