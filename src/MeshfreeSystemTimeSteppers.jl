
# Default case: Everything is decoupled
export RelaxationStepper, ARS2IMEX

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

# In your MeshfreeTimeSteppers.jl (or a new SystemTimeSteppers.jl)
# Ensure necessary using statements:
# using ..ParticleGrids, ..Interpolations, ..SimSettings
# using ..ScalarHyperbolicEquations, ..SystemHyperbolicEquations # Or your actual module names

# Placeholder for system equation type if not defined elsewhere accessible from here
# You should use your actual definitions.
# abstract type AbstractSystemHyperbolicEquations end 
# struct DiagonalLinearAdvectionSystemExample <: AbstractSystemHyperbolicEquations
#     speeds::Vector{Float64}
# end
# num_components(eq::DiagonalLinearAdvectionSystemExample) = length(eq.speeds)

# In a new file like ImplicitSolvers.jl or within MeshfreeTimeSteppers.jl


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
} <: TimeIntegration.TimeStepper # Added TimeIntegration.

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

function TimeIntegration.initTimeStepper(
    ars2::ARS2IMEX,
    system_pg::Vector{<:ParticleGrids.ParticleGrid}, # Added ParticleGrids.
    settings::SimSettings.SimSetting # Added SimSettings.
)
    if isempty(system_pg) return end
    for k_comp in 1:length(system_pg)
        scalar_grid_k = system_pg[k_comp]
        TimeIntegration.initTimeStep(ars2.gradientInterpolator, scalar_grid_k, settings.interpAlpha, settings.interpRange)
        if !isnothing(ars2.fallbackInterpolator)
            TimeIntegration.initTimeStep(ars2.fallbackInterpolator, scalar_grid_k, settings.interpAlpha, settings.interpRange)
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