
# Default case: Everything is decoupled
export RelaxationStepper, ARS2IMEX, GeneralIMEXTimeStepper, ARS233, PareschiRussoIMEXSSP3, ARS222, SSP2332

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
    is_first_mood_stage_in_rk::Bool
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
        for p_idx_cv in 1:N_particles; scalar_grid_k.grid[p_idx_cv].rho = U_state_k_view[p_idx_cv]; end
        
        copyCurvatures!(scalar_grid_k) 

        for p_idx in 1:N_particles
            div_high_k_p = gradientInterpolator(
                scalar_grid_k, p_idx, U_state_k_view, scalar_eq_k, settings; setCurvature=true
            )
            
            rho_candidate_for_mood = U_state_k_view[p_idx] - dt_for_mood_check * div_high_k_p
            K_E_out_sys[p_idx, k_comp] = -div_high_k_p

            if !isnothing(fallbackInterpolator) && mood_criterion(scalar_grid_k, p_idx, U_state_k_view, rho_candidate_for_mood; firstStage=is_first_mood_stage_in_rk)
                div_fallback_k_p = fallbackInterpolator(
                    scalar_grid_k, p_idx, U_state_k_view, scalar_eq_k, settings; setCurvature=false
                )
                K_E_out_sys[p_idx, k_comp] = -div_fallback_k_p
            end
        end
        for p_idx_cv in 1:N_particles; scalar_grid_k.grid[p_idx_cv].rho = temp_rho_backup_k[p_idx_cv]; end
    end
end

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
        for p_idx in 1:N_particles
            ars2.U_n_sys[p_idx, k_comp] = system_pg[k_comp].grid[p_idx].rho
        end
    end

    time_s1_implicit_eval = time_n + dt * gamma
    ars2.U_stage1_sys .= ars2.U_n_sys 
    
    u_particle_iter_buffer = Vector{Float64}(undef, N_components)
    rhs_const_buffer       = Vector{Float64}(undef, N_components)

    for p_idx in 1:N_particles
        particle_pos = system_pg[1].grid[p_idx].pos 
        u_particle_iter_buffer .= @view ars2.U_n_sys[p_idx, :] 
        rhs_const_buffer       .= @view ars2.U_n_sys[p_idx, :] 
        
        solve!(ars2.implicit_solver, # Added ImplicitSolvers.
            u_particle_iter_buffer, rhs_const_buffer, dt * gamma,
            ars2.source_term, particle_pos, time_s1_implicit_eval, N_components
        )
        ars2.U_stage1_sys[p_idx, :] .= u_particle_iter_buffer
    end

    for p_idx in 1:N_particles
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

    for p_idx in 1:N_particles
        for k_comp in 1:N_components
            ars2.U_temp_sys[p_idx, k_comp] = ars2.U_n_sys[p_idx, k_comp] + 
                                            dt * (1.0 - 2.0*gamma) * ars2.K_E1_sys[p_idx, k_comp] +
                                            dt * gamma * ars2.S_U_stage1_sys[p_idx, k_comp]
        end
    end

    time_s2_implicit_eval = time_n + dt 
    ars2.U_stage2_sys .= ars2.U_temp_sys

    for p_idx in 1:N_particles
        particle_pos = system_pg[1].grid[p_idx].pos
        u_particle_iter_buffer .= @view ars2.U_temp_sys[p_idx, :]
        rhs_const_s2_view = @view ars2.U_temp_sys[p_idx, :]
        
        ImplicitSolvers.solve!(ars2.implicit_solver, # Added ImplicitSolvers.
            u_particle_iter_buffer, rhs_const_s2_view, dt * gamma,
            ars2.source_term, particle_pos, time_s2_implicit_eval, N_components
        )
        ars2.U_stage2_sys[p_idx, :] .= u_particle_iter_buffer
    end

    for p_idx in 1:N_particles
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

    for p_idx in 1:N_particles
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

# In MeshfreeTimeSteppers.jl or SystemIMEXTimeSteppers.jl or TimeIntegration.jl
# (Ensure all necessary `using` statements for types from other modules are present)

# struct GeneralIMEXTimeStepper{...} <: TimeIntegration.TimeStepper
#   ... (fields as defined before) ...
# end

# function TimeIntegration.initTimeStepper(imex_ts::GeneralIMEXTimeStepper, ...)
#   ... (as defined before) ...
# end

# --- Corrected Functor for GeneralIMEXTimeStepper ---
function (imex_ts::GeneralIMEXTimeStepper)(
        scalar_equations::Vector{<:ScalarHyperbolicEquations.ScalarHyperbolicEquation},
        system_pg::Vector{<:ParticleGrids.ParticleGrid},
        settings::SimSettings.SimSetting,
        time_n::Real,
        dt::Real
    )

    N_particles = length(system_pg[1].grid)
    N_components = length(scalar_equations)
    s = imex_ts.num_stages
    bt = imex_ts.butcher_tableau # A (implicit), At (explicit), c (implicit_times), ct (explicit_times), b (weights)

    if size(imex_ts.U_n_sys,1) != N_particles || size(imex_ts.U_n_sys,2) != N_components
        error("GeneralIMEXTimeStepper buffers not sized correctly. Expected ($(N_particles)x$(N_components)). Re-initialize instance.")
    end

    # --- 0. Store U^n from system_pg ---
    for k_comp in 1:N_components
        for p_idx in 1:N_particles
            imex_ts.U_n_sys[p_idx, k_comp] = system_pg[k_comp].grid[p_idx].rho
        end
    end

    # Temporary particle-local vectors for implicit solve, reused across particles/stages
    u_particle_iter_buffer = Vector{Float64}(undef, N_components)
    # rhs_for_implicit_solve_particle was the problematic variable name
    # Let's use a clear name for the RHS of Y_i - coeff*S(Y_i) = RHS_FORMULA
    Y_i_base_particle = Vector{Float64}(undef, N_components)


    # --- Loop through stages i = 1 to s ---
    for i in 1:s
        # current_Y_i_sys is an alias to imex_ts.Y_stages_sys[i]
        # It will store the fully computed Y_i for the current stage.
        current_Y_i_sys = imex_ts.Y_stages_sys[i]
        
        # Initialize Y_i_base = U^n for this stage's calculation
        # This Y_i_base will accumulate U^n + explicit_sum + implicit_sum_prev
        # (Note: Y_stages_sys[i] is being used as Y_i_base here before implicit solve)
        current_Y_i_sys .= imex_ts.U_n_sys # Start with U^n

        # Calculate explicit sum part for Y_i: Sum_E = dt * sum_{j=1}^{i-1} At[i,j] * K_E_stages_sys[j]
        for j in 1:(i-1)
            if bt.At[i,j] != 0.0
                for p_idx_loop in 1:N_particles, k_comp_loop in 1:N_components
                    current_Y_i_sys[p_idx_loop, k_comp_loop] += dt * bt.At[i,j] * imex_ts.K_E_stages_sys[j][p_idx_loop, k_comp_loop]
                end
            end
        end

        # Calculate implicit sum from previous stages: Sum_I_prev = dt * sum_{j=1}^{i-1} A[i,j] * K_I_stages_sys[j]
        for j in 1:(i-1)
            if bt.A[i,j] != 0.0
                for p_idx_loop in 1:N_particles, k_comp_loop in 1:N_components
                    current_Y_i_sys[p_idx_loop, k_comp_loop] += dt * bt.A[i,j] * imex_ts.K_I_stages_sys[j][p_idx_loop, k_comp_loop]
                end
            end
        end
        
        # current_Y_i_sys now holds U^n + Sum_E + Sum_I_prev, which is the RHS for the implicit solve part:
        # Y_i - dt * A[i,i] * F_I(Y_i, t_n + c[i]*dt) = current_Y_i_sys_before_solve
        
        if abs(bt.A[i,i]) > 1e-14 # If stage i is implicitly dependent on F_I(Y_i)
            time_implicit_eval = time_n + bt.c[i] * dt
            
            for p_idx in 1:N_particles
                particle_pos = system_pg[1].grid[p_idx].pos
                
                # Initial guess for Y_i for this particle is what's in current_Y_i_sys
                u_particle_iter_buffer .= @view current_Y_i_sys[p_idx, :] 
                # The constant part for the solver is also what's currently in current_Y_i_sys
                Y_i_base_particle      .= @view current_Y_i_sys[p_idx, :] 
                                          
                ImplicitSolvers.solve!(imex_ts.implicit_solver,
                    u_particle_iter_buffer,  # Initial guess & output for Y_i at this particle
                    Y_i_base_particle,       # Base for the solve: U^n + Sum_E + Sum_I_prev
                    dt * bt.A[i,i],          # dt_coefficient_for_S = dt * a_ii
                    imex_ts.source_term_object,
                    particle_pos, time_implicit_eval, N_components
                )
                current_Y_i_sys[p_idx, :] .= u_particle_iter_buffer # Store solved Y_i back
            end
        end
        # If A[i,i] == 0, then current_Y_i_sys (which is imex_ts.Y_stages_sys[i]) 
        # already holds the final Y_i for this stage.

        # Evaluate and store K_Ei = F_E(Y_i, t_n + ct[i]*dt)
        # The explicit tendency is evaluated using the fully formed Y_i (current_Y_i_sys) from this stage.
        time_explicit_eval = time_n + bt.ct[i] * dt
        compute_explicit_tendency_with_mood!( # Ensure this helper is accessible
            imex_ts.K_E_stages_sys[i], current_Y_i_sys, 
            imex_ts.gradientInterpolator, imex_ts.fallbackInterpolator, imex_ts.mood,
            scalar_equations, system_pg, settings, dt, (i==1) 
        )

        # Evaluate and store K_Ii = F_I(Y_i, t_n + c[i]*dt)
        time_implicit_eval_for_KI = time_n + bt.c[i] * dt 
        for p_idx in 1:N_particles
            particle_pos = system_pg[1].grid[p_idx].pos
            Y_i_p_view = @view current_Y_i_sys[p_idx, :]       # Input is the solved Y_i
            K_Ii_p_view = @view imex_ts.K_I_stages_sys[i][p_idx, :] # Output buffer
            imex_ts.source_term_object(K_Ii_p_view, Y_i_p_view, particle_pos, time_implicit_eval_for_KI)
        end
    end # End of stages loop

    # --- Final Update ---
    # U^{n+1} = U^n + dt * sum_{i=1 to s} (b_i * K_Ei + b_i * K_Ii)
    U_np1_sys_temp = copy(imex_ts.U_n_sys) 

    for i in 1:s
        if bt.b[i] != 0.0 
            for p_idx_loop in 1:N_particles, k_comp_loop in 1:N_components
                U_np1_sys_temp[p_idx_loop, k_comp_loop] += 
                    dt * (bt.bt[i] * imex_ts.K_E_stages_sys[i][p_idx_loop, k_comp_loop] + 
                           bt.b[i] * imex_ts.K_I_stages_sys[i][p_idx_loop, k_comp_loop])
            end
        end
    end

    # Update physical particleGrid
    for k_comp in 1:N_components
        for p_idx in 1:N_particles
            system_pg[k_comp].grid[p_idx].rho = U_np1_sys_temp[p_idx, k_comp]
        end
    end
    
    for k_comp in 1:N_components
        for p_obj in system_pg[k_comp].grid
            p_obj.moodEvent = false 
        end
    end
end



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