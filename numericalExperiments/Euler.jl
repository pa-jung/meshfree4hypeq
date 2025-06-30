# File: euler1D_Makie.jl

# --- Module Imports ---
using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration 
using Meshfree4ScalarEq.Interpolations 
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq.FluxFunctions
using Meshfree4ScalarEq.SourceTerms 
using Meshfree4ScalarEq.ImplicitSolvers 
using Meshfree4ScalarEq.PlottingUtils # If calculateStats is here
using Random
using IPlotPDESols
using Meshfree4ScalarEq # For SEED, rng

# --- 1D Euler System Specifics ---
const GAS_GAMMA_EULER = 1.4 # Adiabatic index

"""
    pressure_from_euler_conserved(rho, m, E)

Calculates pressure from conserved variables for 1D Euler.
"""
function pressure_from_euler_conserved(rho::Real, m::Real, E::Real)::Float64
    if rho < 1e-9 # Density floor
        # @warn "Density rho = $rho < 1e-9, flooring pressure calculation."
        return 1e-9 # Avoid division by zero or negative pressure
    end
    # E = p/(gamma-1) + 0.5*rho*u^2 = p/(gamma-1) + 0.5*m^2/rho
    pressure = (GAS_GAMMA_EULER - 1.0) * (E - 0.5 * m^2 / rho)
    return max(pressure, 1e-9) # Pressure floor
end

"""
    euler1D_physical_fluxes(rho, m, E)

Returns the physical flux vector [F_rho, F_m, F_E] for 1D Euler.
U = [rho, m, E]
F(U) = [m, m^2/rho + p, (E+p)m/rho]
"""
function euler1D_physical_fluxes(rho::Real, m::Real, E::Real)::NTuple{3, Float64}
    if rho < 1e-9 # Density floor for safety
        # @warn "Density rho = $rho < 1e-9 in flux calculation."
        return (0.0, pressure_from_euler_conserved(1e-9,0.0,0.0), 0.0) # Return flux at some floor state
    end
    ux = m / rho
    p = pressure_from_euler_conserved(rho, m, E)

    F1 = m
    F2 = m * ux + p
    F3 = (E + p) * ux
    return (F1, F2, F3)
end

# --- Initial Condition Functions for 3 Macroscopic Variables (rho, m, E) ---
# These return (rho_val, m_val, E_val)

"""
Smooth Gaussian initial condition for 1D Euler (rho, velocity u, pressure p).
params: NamedTuple e.g., (
    rho_spec=(amp, mean, width, offset), 
    u_spec=(amp, mean, width, offset), 
    p_spec=(amp, mean, width, offset)
)
Returns (rho, rho*u, E)
"""
function eulerSmooth1DInit(x::Real, params::NamedTuple)::NTuple{3, Float64}
    rho_s, u_s, p_s = params.rho_spec, params.u_spec, params.p_spec

    rho_val = rho_s.off + rho_s.amp * exp(-((x - rho_s.mean) / rho_s.width)^2)
    u_val   = u_s.off   + u_s.amp   * exp(-((x - u_s.mean) / u_s.width)^2)
    p_val   = p_s.off   + p_s.amp   * exp(-((x - p_s.mean) / p_s.width)^2)

    rho_val = max(rho_val, 1e-6) # Density floor
    p_val   = max(p_val, 1e-6)   # Pressure floor

    m_val = rho_val * u_val
    E_val = p_val / (GAS_GAMMA_EULER - 1.0) + 0.5 * rho_val * u_val^2

    return (rho_val, m_val, E_val)
end

"""
1D Shock Tube (Riemann problem in x) for Euler variables.
params: NamedTuple e.g. (stateL=(rho,u,p), stateR=(rho,u,p), shock_pos_x=0.0)
Returns conserved (rho, m, E).
"""
function eulerShockTube1DInit(x::Real, params::NamedTuple)::NTuple{3, Float64}
    stateL_prim, stateR_prim, shock_pos_x = params.stateL, params.stateR, params.shock_pos_x
    
    rho_val, u_val, p_val = x < shock_pos_x ? stateL_prim : stateR_prim

    rho_val = max(rho_val, 1e-6)
    p_val   = max(p_val, 1e-6)

    m_val = rho_val * u_val
    E_val = p_val / (GAS_GAMMA_EULER - 1.0) + 0.5 * rho_val * u_val^2

    return (rho_val, m_val, E_val)
end
# --- NEW: Analytical Riemann Solver for 1D Euler Equations ---

"""
    eulerShockTube1DAnalytic(x::Real, t::Real, params::NamedTuple)

Provides the exact solution to the 1D Euler Riemann problem (shock tube).
The solution is self-similar and depends on the coordinate s = (x - x0)/t.

# Arguments
- `x`: Spatial coordinate.
- `t`: Time.
- `params`: A NamedTuple containing the initial conditions, e.g.,
  `(stateL_prim=(rhoL, uL, pL), stateR_prim=(rhoR, uR, pR), shock_pos_x=0.0)`
  The gas gamma is assumed to be the global `GAS_GAMMA_EULER`.

# Returns
- `NTuple{3, Float64}`: The conserved variables (rho, m, E) at (x,t).
"""
function eulerShockTube1DAnalytic(x::Real, t::Real, params::NamedTuple)::NTuple{3, Float64}
    
    # --- 1. Extract Initial States and Parameters ---
    stateL_prim = params.stateL # (rho_L, u_L, p_L)
    stateR_prim = params.stateR # (rho_R, u_R, p_R)
    x0 = params.shock_pos_x          # Initial discontinuity position
    gamma = GAS_GAMMA_EULER

    rho_L, u_L, p_L = stateL_prim
    rho_R, u_R, p_R = stateR_prim

    # Check for vacuum generation, which this solver doesn't handle
    if (2.0 / (gamma - 1.0)) * (sqrt(gamma*p_L/rho_L) + sqrt(gamma*p_R/rho_R)) <= (u_R - u_L)
        @warn "Vacuum is generated for these initial conditions. Analytical solver may fail."
    end
    
    # Return initial condition for t=0
    if t <= 1e-9
        return eulerShockTube1DInit(x, params)
    end

    # --- 2. Solve for Pressure in the Star Region (p_star) ---
    # This is the core of the Riemann solver. We need to find the root of the equation:
    # f(p, p_side, rho_side, u_side) + f(p, p_other_side, ...) + (u_R - u_L) = 0
    # where f describes the velocity change across the left/right waves.
    
    c_L = sqrt(gamma * p_L / rho_L) # Sound speed in left state
    c_R = sqrt(gamma * p_R / rho_R) # Sound speed in right state
    
    # Function whose root gives p_star
    function pressure_func(p_star_guess::Real)
        # Left wave (shock or rarefaction)
        f_L = 0.0
        if p_star_guess > p_L # Left shock
            A_L = 2.0 / ((gamma + 1.0) * rho_L)
            B_L = p_L * (gamma - 1.0) / (gamma + 1.0)
            f_L = (p_star_guess - p_L) * sqrt(A_L / (p_star_guess + B_L))
        else # Left rarefaction
            f_L = (2.0 * c_L / (gamma - 1.0)) * ((p_star_guess / p_L)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        end
        
        # Right wave (shock or rarefaction)
        f_R = 0.0
        if p_star_guess > p_R # Right shock
            A_R = 2.0 / ((gamma + 1.0) * rho_R)
            B_R = p_R * (gamma - 1.0) / (gamma + 1.0)
            f_R = (p_star_guess - p_R) * sqrt(A_R / (p_star_guess + B_R))
        else # Right rarefaction
            f_R = (2.0 * c_R / (gamma - 1.0)) * ((p_star_guess / p_R)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        end
        
        return f_L + f_R + (u_R - u_L)
    end

    # Iterative root-finding for p_star (e.g., Newton-Raphson or a bracketing method)
    # For simplicity, we use a basic iterative solver here. A robust library like Roots.jl is better.
    p_star = 0.5 * (p_L + p_R) # Initial guess
    p_min_guess = min(p_L, p_R) * 1e-2
    p_max_guess = max(p_L, p_R) * 1e2
    
    # Simple bisection/secant-like method
    for _ in 1:100 # Max iterations
        f_p = pressure_func(p_star)
        if abs(f_p) < 1e-9; break; end
        
        # Simple update logic (can be improved with Newton's method)
        # This is a basic secant/newton step approximation
        dfdp = (pressure_func(p_star * 1.001) - f_p) / (p_star * 0.001)
        p_star -= f_p / (dfdp + 1e-9) # Avoid division by zero
        if p_star < 0; p_star = 1e-9; end # Enforce positivity
    end

    # --- 3. Calculate Star Region Velocity (u_star) and Wave Speeds ---
    f_L_final = 0.0
    if p_star > p_L; f_L_final = (p_star - p_L) * sqrt((2.0/((gamma+1.0)*rho_L)) / (p_star + p_L*(gamma-1.0)/(gamma+1.0)));
    else; f_L_final = (2.0*c_L/(gamma-1.0)) * ((p_star/p_L)^((gamma-1.0)/(2.0*gamma)) - 1.0); end
    u_star = 0.5 * (u_L + u_R) + 0.5 * (pressure_func(p_star) - f_L_final - f_L_final) # This is not quite right
    u_star = u_L - f_L_final # Correct way to find u_star from left wave

    # --- 4. Determine Wave Speeds and Regions ---
    local rho_star_L, rho_star_R
    local S_L, S_R # Left and Right wave speeds
    
    # Left Wave
    if p_star > p_L # Left Shock
        S_L = u_L - c_L * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_L) + (gamma - 1.0) / (2.0 * gamma))
        rho_star_L = rho_L * ((p_star / p_L) + (gamma - 1.0) / (gamma + 1.0)) / (1.0 + (p_star / p_L) * (gamma - 1.0) / (gamma + 1.0))
    else # Left Rarefaction
        S_rarefaction_head_L = u_L - c_L
        c_star_L = c_L * (p_star / p_L)^((gamma - 1.0) / (2.0 * gamma))
        S_rarefaction_tail_L = u_star - c_star_L
        rho_star_L = rho_L * (p_star / p_L)^(1.0 / gamma)
    end

    # Right Wave
    if p_star > p_R # Right Shock
        S_R = u_R + c_R * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_R) + (gamma - 1.0) / (2.0 * gamma))
        rho_star_R = rho_R * ((p_star / p_R) + (gamma - 1.0) / (gamma + 1.0)) / (1.0 + (p_star / p_R) * (gamma - 1.0) / (gamma + 1.0))
    else # Right Rarefaction
        S_rarefaction_head_R = u_R + c_R
        c_star_R = c_R * (p_star / p_R)^((gamma - 1.0) / (2.0 * gamma))
        S_rarefaction_tail_R = u_star + c_star_R
        rho_star_R = rho_R * (p_star / p_R)^(1.0 / gamma)
    end

    S_contact = u_star # Speed of the contact discontinuity

    # --- 5. Find Solution at Query Point (x,t) ---
    s_query = (x - x0) / t # Self-similar coordinate

    rho_final, u_final, p_final = 0.0, 0.0, 0.0

    if s_query <= S_contact # Left of contact
        if p_star > p_L # Left Shock
            if s_query <= S_L
                rho_final, u_final, p_final = rho_L, u_L, p_L
            else
                rho_final, u_final, p_final = rho_star_L, u_star, p_star
            end
        else # Left Rarefaction
            if s_query <= S_rarefaction_head_L
                rho_final, u_final, p_final = rho_L, u_L, p_L
            elseif s_query >= S_rarefaction_tail_L
                rho_final, u_final, p_final = rho_star_L, u_star, p_star
            else # Inside rarefaction fan
                u_final = (2.0 / (gamma + 1.0)) * (c_L + (gamma - 1.0) / 2.0 * u_L + s_query)
                c_final = c_L - (gamma - 1.0) / 2.0 * (u_final - u_L)
                rho_final = rho_L * (c_final / c_L)^(2.0 / (gamma - 1.0))
                p_final = p_L * (rho_final / rho_L)^gamma
            end
        end
    else # Right of contact (s_query > S_contact)
        if p_star > p_R # Right Shock
            if s_query >= S_R
                rho_final, u_final, p_final = rho_R, u_R, p_R
            else
                rho_final, u_final, p_final = rho_star_R, u_star, p_star
            end
        else # Right Rarefaction
            if s_query >= S_rarefaction_head_R
                rho_final, u_final, p_final = rho_R, u_R, p_R
            elseif s_query <= S_rarefaction_tail_R
                rho_final, u_final, p_final = rho_star_R, u_star, p_star
            else # Inside rarefaction fan
                u_final = (2.0 / (gamma + 1.0)) * (-c_R + (gamma - 1.0) / 2.0 * u_R + s_query)
                c_final = c_R + (gamma - 1.0) / 2.0 * (u_R - u_final)
                rho_final = rho_R * (c_final / c_R)^(2.0 / (gamma - 1.0))
                p_final = p_R * (rho_final / rho_R)^gamma
            end
        end
    end

    # --- 6. Convert final primitive variables to conserved variables ---
    m_final = rho_final * u_final
    E_final = p_final / (gamma - 1.0) + 0.5 * rho_final * u_final^2
    
    return (rho_final, m_final, E_final)
end
# --- Analytical Solution Placeholder ---
function eulerSystemAnalytic_dummy(x::Real, t::Real, init_func::Function, init_params_tuple, N_macro_vars::Int)
    if t == 0.0
        # For 2D IC functions that take (x,y,params)
        if applicable(init_func, x, 0.0, init_params_tuple) 
            return init_func(x, 0.0, init_params_tuple) # Call with a dummy y=0.0
        else # For 1D IC functions that take (x,params)
            return init_func(x, init_params_tuple)
        end
    else
        return NTuple{N_macro_vars, Float64}(NaN for _ in 1:N_macro_vars)
    end
end


# --- Main Simulation Runner for 1D Euler Relaxation System ---
function RunSystem1DEulerSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    println("\n--- Running 1D Euler System Simulation (Relaxation Method) ---")
    run_params = copy(params)
    rng_state_backup = copy(Meshfree4ScalarEq.rng)
    local sim_data_result = nothing

    try
        tmax::Float64 = run_params["tmax"]
        N_particles::Int = run_params["N"]

        xmin = run_params["xmin"]
        xmax = run_params["xmax"]

        initFunc_name::String = run_params["init_func"]
        init_params_tuple = run_params["init_params"] 

        cfl_val = get(run_params, "CFL", nothing)
        dt_val = get(run_params, "dt", nothing)
        save_freq::Int = run_params["save_frequency"]
        interp_alpha::Float64 = run_params["interp_alpha"]
        interp_range_factor::Float64 = run_params["interp_range"]
        randomness_factor::Float64 = run_params["randomness_factor"]

        timestepper_name = run_params["timestepper"] # e.g., "ARS2IMEX_Relax"
        main_grad_name = get(run_params,"main_gradient",nothing)
        muscl_order_param = get(run_params,"order",nothing)
        main_flux_name = get(run_params,"main_flux", nothing)
        
        mood_name = get(run_params, "MOOD", nothing)
        delta_relax = get(run_params, "delta_relax", nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing)
        fallback_flux_name = get(run_params,"fallback_flux", nothing)

        # Relaxation Velocities: Vector of Tuples, one pair for each macro var
        relax_velocities_config = get(run_params,"relax_velocities", nothing)
        relax_epsilon_val = get(run_params,"relax_epsilon", nothing)

        N_macro_vars = length(relax_velocities_config) # rho, m, E
        # if length(relax_velocities_pairs_list) != N_macro_vars
        #     error("`relax_velocities` must provide a pair of speeds for each of $N_macro_vars macroscopic variables.")
        # end

        println("  System Timestepper: $(timestepper_name), Main Gradient: $(main_grad_name)")
        println("  IC: $(initFunc_name), N_particles: $(N_particles), Domain: [$xmin,$xmax]")
                # --- Base 1D Grid for Geometry ---
        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / N_particles
        local base_particleGrid1D

        randomness = randomness_factor * dx_nominal
        base_particleGrid1D = ParticleGrid1D(xmin, xmax, N_particles; randomness = randomness)
        interp_range = interp_range_factor * base_particleGrid1D.dx

        # --- Determine dt ---
        local actual_dt::Float64
        if !isnothing(cfl_val)
            max_abs_kinetic_speed = 0.0
            for speeds in relax_velocities_config
                max_abs_kinetic_speed = max(max_abs_kinetic_speed, maximum(abs.(speeds)))
            end
            if max_abs_kinetic_speed < 1e-9; max_abs_kinetic_speed = 1.0; end
            
            temp_eq_for_dt = LinearAdvection(max_abs_kinetic_speed)
            actual_dt = cfl_val * getTimeStep(base_particleGrid1D, temp_eq_for_dt, interp_alpha, interp_range)
        elseif !isnothing(dt_val); actual_dt = dt_val;
        else error("Either CFL or dt must be specified."); end
        println("  Calculated/Used dt: $actual_dt")
        if timestepper_name != "Analytic"



            # --- Build Scalar Method Components ---
            MainFlux = if main_flux_name == "Rusanov"; RusanovFlux() else error("Flux $main_flux_name NYI"); end
            FallbackFlux = if fallback_flux_name == "Rusanov"; RusanovFlux() else error("Flux $fallback_flux_name NYI"); end
            
            MainGrad = if main_grad_name == "MUSCL"
                        MUSCL(muscl_order_param-1; numericalFlux=MainFlux, weightFunction=exponentialWeightFunction())
                        elseif main_grad_name == "WENO"
                            WENO(order)
                        elseif main_grad_name == "MUSCLlimit"
                            MUSCLlimited(1; numericalFlux=MainFlux, weightFunction=exponentialWeightFunction())
                    elseif main_grad_name == "Upwind"
                        UpwindGradient(1; numericalFlux=MainFlux, algType="Classic", weightFunction=exponentialWeightFunction())
                    elseif isnothing(main_grad_name)
                            @assert timestepper_name == "Analytic"
                    else error("Unknown MainGrad: $main_grad_name"); end

            FallbackGrad =  if fallback_grad_name == "Upwind"
                                UpwindGradient(1; numericalFlux=FallbackFlux, algType="Classic", weightFunction=exponentialWeightFunction())
                            elseif !isnothing(fallback_grad_name)
                                error("Only Upwind Gradient possible as fallback!")
                            end   
            # --- Build Method Components (Ensure types/constructors are accessible) ---
            local mood_fun
            if mood_name == "U1"; mood_fun = MOODu1(deltaRelax = delta_relax)
            elseif mood_name == "U2"; mood_fun = MOODu2(deltaRelax = delta_relax)
            elseif mood_name == "LoubertU2"; mood_fun = MOODLoubertU2(deltaRelax = delta_relax)
            elseif mood_name == "none"; mood_fun = NoMOOD()
            elseif mood_name == "only"; mood_fun = OnlyMOOD()
            elseif mood_name == "firstNone"; mood_fun = FirstStageNoMOOD()
            elseif mood_name == "alt"; mood_fun = MOODAlt(deltaRelax = delta_relax)
            elseif !isnothing(mood_name); error("MOOD '$mood_name' not recognized") end
            # --- REFINED: Setup for Kinetic System using relax_velocities_config ---
            num_kinetic_components_per_macro_var = [length(speed_group) for speed_group in relax_velocities_config]
            N_total_kinetic_components = sum(num_kinetic_components_per_macro_var)

            kinetic_to_macro_map = Vector{Vector{Int}}(undef, N_macro_vars)
            current_kinetic_idx_offset = 0
            for i_macro_map in 1:N_macro_vars
                num_kin_for_this_macro = num_kinetic_components_per_macro_var[i_macro_map]
                kinetic_to_macro_map[i_macro_map] = collect((current_kinetic_idx_offset + 1) : (current_kinetic_idx_offset + num_kin_for_this_macro))
                current_kinetic_idx_offset += num_kin_for_this_macro
            end
            
            kinetic_eqs = Vector{LinearAdvection{Float64}}(undef, N_total_kinetic_components)
            M_funcs = Vector{Function}(undef, N_total_kinetic_components)
            
            global_kinetic_idx_counter = 1
            for i_macro_phys in 1:N_macro_vars # Iterate through physical macroscopic variables (1=rho, 2=m, 3=E)
                speeds_for_this_macro_var = relax_velocities_config[i_macro_phys]
                for speed_val in speeds_for_this_macro_var 
                    kinetic_eqs[global_kinetic_idx_counter] = LinearAdvection(Float64(speed_val))
                    
                    # Maxwellian M_k takes the full macroscopic state vector (rho, m, E) as splatted arguments
                    # and uses i_macro_phys to pick the correct component of U_macro and F_macro.
                    M_funcs[global_kinetic_idx_counter] = 
                        (U_macro_args::Vararg{Float64}) -> begin # U_macro_args will be (rho_p, m_p, E_p)
                            # U_macro_args is already a tuple of the macro values
                            F_macro_vector_at_p = euler1D_physical_fluxes(U_macro_args...) # Splat into fluxes
                            
                            U_macro_component_val = U_macro_args[i_macro_phys]
                            F_macro_component_val = F_macro_vector_at_p[i_macro_phys]
                            
                            current_relax_speed = Float64(speed_val)
                            if abs(current_relax_speed) < 1e-12; current_relax_speed = sign(current_relax_speed + 1e-13) * 1e-12; end

                            return 0.5 * (U_macro_component_val + F_macro_component_val / current_relax_speed)
                        end
                    global_kinetic_idx_counter += 1
                end
            end
            
            relaxation_source = SourceTerms.RelaxationSourceTerm(M_funcs, relax_epsilon_val, kinetic_to_macro_map)
            implicit_solver = ImplicitSolvers.LinearizedRelaxationImplicitSolver()

            component_pgs = [deepcopy(base_particleGrid1D) for _ in 1:N_total_kinetic_components]

            # --- REFINED: Initialize Macroscopic State first, then Kinetic Component Grids using M_funcs and map ---
            init_func_evaluator = if initFunc_name == "eulerSmooth1D"; eulerSmooth1DInit
                                elseif initFunc_name == "eulerShockTube1D"; eulerShockTube1DInit
                                else error("Unknown system IC name: $initFunc_name for 1D Euler"); end
            
            # This matrix stores the macroscopic IC [rho_0(xp), m_0(xp), E_0(xp)] for each particle
            macro_IC_at_points = Matrix{Float64}(undef, N_particles, N_macro_vars)
            for p_idx in 1:N_particles
                xp = base_particleGrid1D.grid[p_idx].pos
                U_macro_0_at_p_tuple = init_func_evaluator(xp, init_params_tuple)
                for i_mvar in 1:N_macro_vars
                    macro_IC_at_points[p_idx, i_mvar] = U_macro_0_at_p_tuple[i_mvar]
                end
            end

            # Initialize each kinetic component grid
            for k_global_comp in 1:N_total_kinetic_components
                for p_idx in 1:N_particles
                    # M_funcs[k_global_comp] expects splatted arguments (rho_0, m_0, E_0) for particle p_idx
                    current_macro_ic_for_particle_p = (macro_IC_at_points[p_idx,1], macro_IC_at_points[p_idx,2], macro_IC_at_points[p_idx,3])
                    component_pgs[k_global_comp].grid[p_idx].rho = M_funcs[k_global_comp](current_macro_ic_for_particle_p...)
                end
            end
            
            local system_method_instance::TimeIntegration.TimeStepper
            if timestepper_name == "ARS222" # Default IMEX choice
                system_method_instance = ARS222( MainGrad, FallbackGrad, mood_fun,
                    implicit_solver, relaxation_source,
                    N_particles, N_total_kinetic_components
                )
            elseif timestepper_name == "ARS233" # Default IMEX choice
                system_method_instance = ARS233( MainGrad, FallbackGrad, mood_fun,
                    implicit_solver, relaxation_source,
                    N_particles, N_total_kinetic_components
                )
            elseif timestepper_name == "SSP2" # Default IMEX choice
                system_method_instance = SSP2332( MainGrad, FallbackGrad, mood_fun,
                    implicit_solver, relaxation_source,
                    N_particles, N_total_kinetic_components
                )
            elseif timestepper_name == "SSP3" # Default IMEX choice
                system_method_instance = PareschiRussoIMEXSSP3( MainGrad, FallbackGrad, mood_fun,
                    implicit_solver, relaxation_source,
                    N_particles, N_total_kinetic_components
                )
            else
                error("Unsupported system timestepper for 1D Euler relaxation: $timestepper_name")
            end
            
            sim_settings_obj = SimSetting( # Renamed to avoid conflict with module
                tmax=tmax, dt=actual_dt, interpRange=interp_range, interpAlpha=interp_alpha,
                saveDir="/", 
                saveFreq=save_freq, organiseFiles=false
            )

            println("Starting 1D Euler system time integration...")
            elapsed_time, sys_xs_data, sys_us_data_kinetic, sys_ts_data = mainTimeIntegrator2!(
                system_method_instance, kinetic_eqs, component_pgs, sim_settings_obj
            )
            println("1D Euler system time integration finished in $(round(elapsed_time, digits=2)) seconds.")

            # --- Post-process: Recombine kinetic variables to macroscopic ---
            us_macro_data = Vector{Matrix{Float64}}(undef, length(sys_us_data_kinetic))
            for t_idx in eachindex(sys_us_data_kinetic)
                kinetic_state_matrix_at_t = sys_us_data_kinetic[t_idx] # N_particles x N_total_kinetic_components
                macro_state_matrix_at_t = Matrix{Float64}(undef, N_particles, N_macro_vars)
                
                for p_idx in 1:N_particles
                    for i_macro in 1:N_macro_vars
                        # Get the vector of kinetic indices that sum to this i_macro-th variable
                        indices_for_this_macro = kinetic_to_macro_map[i_macro] # e.g., [1,2] for first macro var
                        
                        # Sum the relevant kinetic components for this particle p_idx
                        # kinetic_state_matrix_at_t[p_idx, indices_for_this_macro] will be a view/vector
                        if isempty(indices_for_this_macro)
                            macro_state_matrix_at_t[p_idx, i_macro] = 0.0
                        else
                            macro_state_matrix_at_t[p_idx, i_macro] = sum(kinetic_state_matrix_at_t[p_idx, idx] for idx in indices_for_this_macro)
                            # Or, more directly if indexing with a vector of indices works as expected for rows/cols:
                            # macro_state_matrix_at_t[p_idx, i_macro] = sum(kinetic_state_matrix_at_t[p_idx, indices_for_this_macro])
                        end
                    end
                end
                us_macro_data[t_idx] = macro_state_matrix_at_t
            end
            sim_data_result = createSimData(sys_xs_data, us_macro_data, sys_ts_data, run_params)
        else
            ts = collect(0:actual_dt:tmax)
            if ts[end] != tmax
                push!(ts, tmax)
            end
            xs = [p.pos for p = base_particleGrid1D.grid]
            us = Vector{Matrix{Float64}}(undef,0)
            for t = ts
                tmp = Matrix(undef, length(xs), 3)
                for (i,x) = enumerate(xs)
                    tmp[i,:] = collect(eulerShockTube1DAnalytic(x,t, init_params_tuple))
                end
                push!(us, tmp)
            end
            sim_data_result = createSimData([xs for _ = ts], us, ts, run_params)
        end
    catch e
         if isa(e, KeyError); @error "Missing required parameter for System Simulation!" key=e.key params=run_params
         else; @error "Error during System Simulation!" params=run_params exception=(e, catch_backtrace()); end
         return nothing
    finally
        copy!(Meshfree4ScalarEq.rng, rng_state_backup)
    end
    return sim_data_result
end

# --- Example SimulationConfig for 1D Euler System ---
euler_smooth_params = (
    rho_spec=(amp=0.1, mean=0.0, width=0.5, off=1.0),
    u_spec  =(amp=0.2, mean=0.0, width=0.5, off=0.5),
    p_spec  =(amp=0.1, mean=0.0, width=0.5, off=1.0)
)

sod_euler_params = ( # Sod shock tube for 1D Euler
    stateL=(rho=1.0, u=0.0, p=1.0),    
    stateR=(rho=0.125, u=0.0, p=0.1), 
    shock_pos_x=0.0 
    # Note: Your plotting range and tmax should be suitable for Sod's problem evolution.
    # Typical Sod domain [-0.5, 0.5], tmax ~ 0.2
)
SEED_value = (:const, Meshfree4ScalarEq.SEED)

sim_config_euler1d_system = SimulationConfig(
    RunSystem1DEulerSimulation, 
    ParamDict(
        "tmax" => 0.2, "N" => 500, 
        "xmin" => -0.5, "xmax" => 1., 
        "CFL" => 0.5, "save_frequency" => 5, 
        "interp_alpha" => 1.0, "interp_range" => 3.5, # Factor for dx
        "init_func" => "eulerShockTube1D", 
        "init_params" => sod_euler_params, 
        "randomness_factor" => 0.2, 
        "SEED" => SEED_value,
        "relax_velocities" => [ (2.0, -2.0), (3.0, -3.0), (4.0, -4.0) ], # Pairs for rho, m, E kinetic components
    ),
    MethodDict( 
        "Slope Limiter" => ParamDict(
            "timestepper" => "ARS222",
            "main_gradient" => "MUSCLlimit", "order" => 2, # MUSCLlimited recon order is 1. this order param is for general MUSCL
            "main_flux" => "Rusanov",
            "MOOD" => "none",
                    "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
            "relax_epsilon" => 1e-6
        ),
        "Regular MOOD" => ParamDict(
            "timestepper" => "ARS222",
            "main_flux" => "Rusanov",
            "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U2", "delta_relax" => false, 
                    "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
            "relax_epsilon" => 1e-6
        ),
        "SSP" => ParamDict(
            "timestepper" => "SSP2",
            "main_flux" => "Rusanov",
                    "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
            "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U2", "delta_relax" => false, 
            "relax_epsilon" => 1e-6
        ),
        "high Order" => ParamDict(
            "timestepper" => "ARS233",
                    "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
            "main_flux" => "Rusanov",
            "main_gradient" => "MUSCL", "order" => 5, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U2", "delta_relax" => false, # More aggressive MOOD
            "relax_epsilon" => 1e-6
        ),
        "Analytic" => ParamDict(
            "timestepper" => "Analytic",
            "randomness_factor" => (:const,0.)
             # No randomness_factor needed when regular=true
        )
    ),
    ["Slope Limiter", "Regular MOOD", "Analytic"]
)

# To run:
show1DSolutionFig(sim_config_euler1d_system) 
# This will require show1DSolutionFig to be adapted to handle SimData1D.u as Vector{Matrix}
# and use the component selector. For now, it will plot the first component (rho_macro).