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
        main_grad_name = run_params["main_gradient"]
        muscl_order_param = run_params["order"]
        main_flux_name = run_params["main_flux"]
        
        mood_name = run_params["MOOD"]
        delta_relax_mood = run_params["delta_relax"]
        fallback_grad_name = run_params["fallback_gradient"]
        fallback_flux_name = run_params["fallback_flux"]

        # Relaxation Velocities: Vector of Tuples, one pair for each macro var
        relax_velocities_config = run_params["relax_velocities"]
        relax_epsilon_val = run_params["relax_epsilon"]

        N_macro_vars = length(relax_velocities_config) # rho, m, E
        # if length(relax_velocities_pairs_list) != N_macro_vars
        #     error("`relax_velocities` must provide a pair of speeds for each of $N_macro_vars macroscopic variables.")
        # end

        println("  System Timestepper: $(timestepper_name), Main Gradient: $(main_grad_name) (MUSCL Recon Order: $(muscl_order_param-1))")
        println("  IC: $(initFunc_name), N_particles: $(N_particles), Domain: [$xmin,$xmax]")

        # --- Base 1D Grid for Geometry ---
        base_particleGrid1D = ParticleGrid1D(xmin, xmax, N_particles; randomness=randomness_factor)
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
                   else error("Unknown MainGrad: $main_grad_name"); end

        FallbackGrad =  if fallback_grad_name == "Upwind"
                            UpwindGradient(1; numericalFlux=FallbackFlux, algType="Classic", weightFunction=exponentialWeightFunction())
                        else 
                            error("Only Upwind Gradient possible as fallback!")
                        end   
        mood_fun = if mood_name == "none"; NoMOOD() elseif mood_name=="U1"; MOODu1(deltaRelax=delta_relax_mood) else error("Unknown MOOD"); end

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
        "tmax" => 0.2, "N" => 100, 
        "xmin" => -0.5, "xmax" => 0.5, 
        "CFL" => 0.5, "save_frequency" => 5, 
        "interp_alpha" => 1.0, "interp_range" => 1.5, # Factor for dx
        "init_func" => "eulerShockTube1D", 
        "init_params" => sod_euler_params, 
        "randomness_factor" => 0.0, 
        "SEED" => SEED_value,
        "N_macro_vars" => 3,
        "order" => 2,
        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
        "MOOD" => "none", # Using MUSCLlimited, so MOOD might be 'none'
        "delta_relax" => false,
        "relax_velocities" => [ (3.0, -3.0), (4.0, -4.0), (5.0, -5.0) ], # Pairs for rho, m, E kinetic components
        "relax_epsilon" => 1e-3 # Larger epsilon might be more stable initially with Picard
    ),
    MethodDict( 
        "Slope Limiter" => ParamDict(
            "timestepper" => "ARS222",
            "main_gradient" => "MUSCLlimit", "order" => 2, # MUSCLlimited recon order is 1. this order param is for general MUSCL
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "relax_epsilon" => 1e-4
        ),
        "Regular MOOD" => ParamDict(
            "timestepper" => "ARS222",
            "main_flux" => "Rusanov",
            "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U1", "delta_relax" => false, # More aggressive MOOD
            "relax_epsilon" => 1e-4
        ),
        "SSP" => ParamDict(
            "timestepper" => "SSP2",
            "main_flux" => "Rusanov",
            "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U1", "delta_relax" => false, # More aggressive MOOD
            "relax_epsilon" => 1e-4
        ),
        "high Order" => ParamDict(
            "timestepper" => "ARS233",
            "main_flux" => "Rusanov",
            "main_gradient" => "MUSCL", "order" => 4, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U1", "delta_relax" => false, # More aggressive MOOD
            "relax_epsilon" => 1e-4
        ),
    ),
    ["Slope Limiter", "Regular MOOD"]; 
    ui_options = Dict("animation_duration_s" => 10., "show_scatter" => false, "system_dimension" => 3) 
    # sys_dim_plot=3 tells plotting to expect 3 macro components (rho,m,E)
)

# To run:
show1DSolutionFig(sim_config_euler1d_system) 
# This will require show1DSolutionFig to be adapted to handle SimData1D.u as Vector{Matrix}
# and use the component selector. For now, it will plot the first component (rho_macro).