# File: euler1D_Makie.jl

# --- Module Imports ---
using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.HyperbolicSystems
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration 
using Meshfree4ScalarEq.Interpolations 
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq.FluxFunctions
using Meshfree4ScalarEq.SourceTerms 
using Meshfree4ScalarEq.ImplicitSolvers 
using Meshfree4ScalarEq.PlottingUtils # If calculateStats is here
using Meshfree4ScalarEq.InitialConditions
using Random
using IPlotPDESols
using Meshfree4ScalarEq # For SEED, rng

# --- Main Simulation Runner for 1D Euler Relaxation System ---
function RunSystem1DEulerSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    println("\n--- Running 1D Euler System Simulation (Relaxation Method) ---")
    run_params = copy(params)
    local sim_data_result = nothing

    try
        tmax::Float64 = run_params["tmax"]
        N_particles::Int = run_params["N"]
        bc::Symbol = run_params["bc"]
        system_name::String = run_params["system"]

        xmin = run_params["xmin"]
        xmax = run_params["xmax"]

        initFunc_name::String = run_params["init_func"]
        init_params_tuple = run_params["init_params"] 

        cfl_val = get(run_params, "CFL", nothing)
        dt_val = get(run_params, "dt", nothing)
        SEED_value = get(run_params, "SEED", nothing)
        snapshots::Int = run_params["snapshots"]
        interp_alpha::Float64 = run_params["interp_alpha"]
        interp_range_factor::Float64 = run_params["interp_range"]
        
        randomness_factor::Float64 = run_params["randomness_factor"]

        timestepper_name = get(run_params,"timestepper",nothing) # e.g., "ARS2IMEX_Relax"
        main_grad_name = get(run_params,"main_gradient",nothing)
        muscl_order_param = get(run_params,"order",nothing)
        main_flux_name = get(run_params,"main_flux", nothing)
        
        mood_name = get(run_params, "MOOD", nothing)
        delta_relax = get(run_params, "delta_relax", nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing)
        fallback_flux_name = get(run_params,"fallback_flux", nothing)

        # Check for Euler
        @assert system_name == "euler" "Only Euler system supported so far!"
        system = EulerEquations()

        # Relaxation Velocities: Vector of Tuples, one pair for each macro var
        relax_velocities_config = get(run_params,"relax_velocities", nothing)
        relax_epsilon_val = get(run_params,"relax_epsilon", nothing)

        N_macro_vars = length(relax_velocities_config) # rho, m, E
        N_ghost::Int64 = bc == :periodic ? 0 : convert(Int64, ceil(interp_range_factor))
        # if length(relax_velocities_pairs_list) != N_macro_vars
        #     error("`relax_velocities` must provide a pair of speeds for each of $N_macro_vars macroscopic variables.")
        # end
        N_total = N_particles + 2*N_ghost
        ic_object = InitialConditions.getInitialCondition(initFunc_name, init_params_tuple)

        println("  System Timestepper: $(timestepper_name), Main Gradient: $(main_grad_name)")
        println("  IC: $(initFunc_name), N_particles: $(N_particles), Domain: [$xmin,$xmax]")
                # --- Base 1D Grid for Geometry ---
        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / N_particles
        local base_particleGrid1D

        rng = MersenneTwister(SEED_value)
        randomness = randomness_factor * dx_nominal
        base_particleGrid1D = ParticleGrid1D(xmin, xmax, N_particles, N_ghost, bc; randomness = randomness, rng = rng)
        interior_indices = base_particleGrid1D.interior_indices
        interp_range = interp_range_factor * base_particleGrid1D.dx
        println(length(base_particleGrid1D.grid))
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

        # --- NEW: Calculate save_frequency in steps ---
        if tmax <= 0 || actual_dt <= 0
            # Handle edge case to avoid division by zero
            save_freq = 1 
        else
            # Calculate the desired time interval between saves
            save_time_interval = tmax / snapshots
            # Convert the time interval to an integer number of steps
            save_frequency_steps = round(Int, save_time_interval / actual_dt)
            # Ensure we always take at least one step before saving
            save_freq = max(1, save_frequency_steps)
        end
        @info "  Calculated/Used dt: $actual_dt"
        if !isnothing(timestepper_name)
            # --- Build Scalar Method Components ---
            MainFlux = if main_flux_name == "Rusanov"; RusanovFlux() else error("Flux $main_flux_name NYI"); end
            FallbackFlux = if fallback_flux_name == "Rusanov"; RusanovFlux() elseif !isnothing(fallback_flux_name); error("Flux $fallback_flux_name NYI"); end
            
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

            
            # This matrix stores the macroscopic IC [rho_0(xp), m_0(xp), E_0(xp)] for each particle
            macro_IC_at_points = Matrix{Float64}(undef, N_particles, N_macro_vars)
            for (i,p_idx) in enumerate(base_particleGrid1D.interior_indices)
                xp = base_particleGrid1D.grid[p_idx].pos
                U_macro_0_at_p_tuple = ic_object(xp)
                for i_mvar in 1:N_macro_vars
                    macro_IC_at_points[i, i_mvar] = U_macro_0_at_p_tuple[i_mvar]
                end
            end

            # Initialize each kinetic component grid
            for k_global_comp in 1:N_total_kinetic_components
                for (i, p_idx) in enumerate(base_particleGrid1D.interior_indices)
                    # M_funcs[k_global_comp] expects splatted arguments (rho_0, m_0, E_0) for particle p_idx
                    current_macro_ic_for_particle_p = (macro_IC_at_points[i,1], macro_IC_at_points[i,2], macro_IC_at_points[i,3])
                    component_pgs[k_global_comp].grid[p_idx].rho = M_funcs[k_global_comp](current_macro_ic_for_particle_p...)
                end
            end
            
            local system_method_instance::TimeIntegration.TimeStepper
            if timestepper_name == "ARS222" # Default IMEX choice
                system_method_instance = ARS222( MainGrad, FallbackGrad, mood_fun,
                    implicit_solver, relaxation_source,
                    N_total, N_total_kinetic_components
                )
            elseif timestepper_name == "ARS233" # Default IMEX choice
                system_method_instance = ARS233( MainGrad, FallbackGrad, mood_fun,
                    implicit_solver, relaxation_source,
                    N_total, N_total_kinetic_components
                )
            elseif timestepper_name == "SSP2" # Default IMEX choice
                system_method_instance = SSP2332( MainGrad, FallbackGrad, mood_fun,
                    implicit_solver, relaxation_source,
                    N_total, N_total_kinetic_components
                )
            elseif timestepper_name == "SSP3" # Default IMEX choice
                system_method_instance = PareschiRussoIMEXSSP3( MainGrad, FallbackGrad, mood_fun,
                    implicit_solver, relaxation_source,
                    N_total, N_total_kinetic_components
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
            ts = collect(0:actual_dt*save_freq:tmax)
            if ts[end] != tmax
                push!(ts, tmax)
            end
            xs = [p.pos for p = base_particleGrid1D.grid[interior_indices]]
            us = Vector{Matrix{Float64}}(undef,0)
            for t = ts
                tmp = Matrix(undef, length(xs), 3)
                for (i,x) = enumerate(xs)
                    tmp[i,:] = collect(ic_object(x,t,system,base_particleGrid1D))
                end
                push!(us, tmp)
            end
            sim_data_result = createSimData([xs for _ = ts], us, ts, run_params)
        end
    catch e
         if isa(e, KeyError); @error "Missing required parameter for System Simulation!" key=e.key params=run_params
         else; @error "Error during System Simulation!" params=run_params exception=(e, catch_backtrace()); end
         return nothing
    end
    return sim_data_result
end

# --- Example SimulationConfig for 1D Euler System ---
euler_smooth_params = (
    (amp=0.1, mean=0.0, width=0.5, off=1.0),
    (amp=0.2, mean=0.0, width=0.5, off=0.5),
    (amp=0.1, mean=0.0, width=0.5, off=1.0)
)

sod_euler_params = ( # Sod shock tube for 1D Euler
    (1.0, 0.0, 1.0),    
    (0.125, 0.0, 0.1), 
    0.0 
    # Note: Your plotting range and tmax should be suitable for Sod's problem evolution.
    # Typical Sod domain [-0.5, 0.5], tmax ~ 0.2
)
SEED_value = 10

sim_config_euler1d_system = SimulationConfig(
    RunSystem1DEulerSimulation, 
    ParamDict(
        "tmax" => 0.2, "N" => 100, "bc" => :outflow,
        "xmin" => -0.5, "xmax" => .5, 
        "CFL" => 0.5, "snapshots" => 5, 
        "interp_alpha" => 1.0, "interp_range" => 1.5, # Factor for dx
        "init_func" => "eulerShockTube",
        "system" => "euler", 
        "init_params" => sod_euler_params, 
        "randomness_factor" => 0., 
        "SEED" => SEED_value,
        "relax_velocities" => [ (2.0, -2.0), (3.0, -3.0), (4.0, -4.0) ], # Pairs for rho, m, E kinetic components
    ),
    MethodDict( 
        "ARS222MUSCL2limiter" => ParamDict(
            "timestepper" => "ARS222",
            "main_gradient" => "MUSCLlimit", "order" => 2, # MUSCLlimited recon order is 1. this order param is for general MUSCL
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "relax_epsilon" => 1e-6
        ),
        "ARS222MUSCL2MOOD" => ParamDict(
            "timestepper" => "ARS222",
            "main_flux" => "Rusanov",
            "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U2", "delta_relax" => 0., 
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
            "relax_epsilon" => 1e-6
        ),
        "ARS222MUSCL2MOOD" => ParamDict(
            "timestepper" => "ARS222",
            "main_flux" => "Rusanov",
            "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U2", "delta_relax" => 0., 
                    "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
            "relax_epsilon" => 1e-6
        ),
        "SSP2MUSCL2MOOD" => ParamDict(
            "timestepper" => "SSP2",
            "main_flux" => "Rusanov",
                    "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
            "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U2", "delta_relax" => 0., 
            "relax_epsilon" => 1e-6
        ),
        "ARS222Upwind" => ParamDict(
            "timestepper" => "ARS222",
            "main_flux" => "Rusanov",
            "main_gradient" => "Upwind", "order" => 1, # MUSCL(1) for 2nd order spatial
            "MOOD" => "none",
            "relax_epsilon" => 1e-6
        ),
        "ARS233MUSCL5MOOD" => ParamDict(
            "timestepper" => "ARS233",
                    "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
            "main_flux" => "Rusanov",
            "main_gradient" => "MUSCL", "order" => 5, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U2", "delta_relax" => 0., # More aggressive MOOD
            "relax_epsilon" => 1e-6
        ),
        "SSP3MUSCL5MOOD" => ParamDict(
            "timestepper" => "SSP3",
                    "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
            "main_flux" => "Rusanov",
            "main_gradient" => "MUSCL", "order" => 5, # MUSCL(1) for 2nd order spatial
            "MOOD" => "U2", "delta_relax" => 0., # More aggressive MOOD
            "relax_epsilon" => 1e-6
        ),
        "Analytic" => ParamDict(
            "randomness_factor" => (:const,0.)
             # No randomness_factor needed when regular=true
        )
    ),
    "Analytic"
)

# To run:
show1DSolutionFig(sim_config_euler1d_system) 
# This will require show1DSolutionFig to be adapted to handle SimData1D.u as Vector{Matrix}
# and use the component selector. For now, it will plot the first component (rho_macro).