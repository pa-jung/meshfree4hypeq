
# File: euler1D_Makie.jl

# --- Module Imports ---
using Meshfree4ScalarEq.HyperbolicPDEs
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration 
using Meshfree4ScalarEq.Interpolations 
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq.FluxFunctions
using Meshfree4ScalarEq.SourceTerms 
using Meshfree4ScalarEq.ImplicitSolvers 
using Meshfree4ScalarEq.InitialConditions
using Random
using IPlotPDESols
              # For RNG state copy


# --- Main Simulation Runner for 1D Relaxation System ---
function runSystem1DSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    @info "\n--- Running 1D Euler System Simulation (Relaxation Method) ---"
    run_params = copy(params)
    local sim_data_result = nothing

    try
        tmax::Float64 = run_params["tmax"]
        N_particles::Int = run_params["N"]
        bc::Symbol = run_params["bc"]
        system_name::String = run_params["PDE"]

        xmin = run_params["xmin"]
        xmax = run_params["xmax"]

        initFunc_name::String = run_params["init_func"]
        init_params_tuple = run_params["init_params"] 

        cfl_val = get(run_params, "CFL", nothing)
        dt_val = get(run_params, "dt", nothing)
        SEED_value = get(run_params, "SEED", nothing)
        snapshots::Int = run_params["snapshots"]
        interp_alpha::Float64 = get(run_params,"interp_alpha", 1.)
        interp_range_factor::Float64 = get(run_params,"interp_range", 1.1)
        
        randomness_factor::Float64 = get(run_params,"randomness_factor",0.)

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
        system = Euler1D()
        N_macro_vars = 3

        # Relaxation Velocities: Vector of Tuples, one pair for each macro var
        relax_velocities_config = get(run_params,"relax_velocities", nothing)
        relax_velocities_config = isnothing(relax_velocities_config) ? [(1., -1.)] : relax_velocities_config
        relax_epsilon_val = get(run_params,"relax_epsilon", nothing)

        N_macro_vars = length(relax_velocities_config) # rho, m, E
        N_ghost::Int64 = bc == :periodic ? 0 : convert(Int64, ceil(interp_range_factor))
        # if length(relax_velocities_pairs_list) != N_macro_vars
        #     error("`relax_velocities` must provide a pair of speeds for each of $N_macro_vars macroscopic variables.")
        # end
        N_total = N_particles + 2*N_ghost
        ic_object = InitialConditions.getInitialCondition(initFunc_name, init_params_tuple)

        @info "  System Timestepper: $(timestepper_name), Main Gradient: $(main_grad_name) IC: $(initFunc_name), N_particles: $(N_particles), Domain: [$xmin,$xmax]"
                # --- Base 1D Grid for Geometry ---
        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / N_particles
        local base_particleGrid1D

        rng = isnothing(SEED_value) ? MersenneTwister(1) : MersenneTwister(SEED_value)
        randomness = randomness_factor * dx_nominal
        base_particleGrid1D = ParticleGrid1D(xmin, xmax, N_particles, N_ghost, bc; randomness = randomness, rng = rng)
        interior_indices = base_particleGrid1D.interior_indices
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
            MainFlux = if main_flux_name == "Rusanov"; RusanovFlux() elseif main_grad_name!="WENO" error("Flux $main_flux_name NYI"); end
            FallbackFlux = if fallback_flux_name == "Rusanov"; RusanovFlux() elseif !isnothing(fallback_flux_name); error("Flux $fallback_flux_name NYI"); end
            
            MainGrad = if main_grad_name == "MUSCL"
                        MUSCL(muscl_order_param-1; numericalFlux=MainFlux, weightFunction=exponentialWeightFunction())
                        elseif main_grad_name == "WENO"
                            WENO(muscl_order_param)
                        elseif main_grad_name == "MUSCLlimit"
                            MUSCLlimited(1; numericalFlux=MainFlux, weightFunction=exponentialWeightFunction())
                    elseif main_grad_name == "Upwind"
                        UpwindGradient(1; numericalFlux=MainFlux, algType="Classic", weightFunction=exponentialWeightFunction())
                    elseif isnothing(main_grad_name)
                            @assert timestepper_name == "Analytic" || timestepper_name == "SimpleSplitting"
                            nothing
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
            kinetic_eqs = Vector{LinearAdvection{2}}(undef, N_total_kinetic)
            M_funcs = Vector{MaxwellianFunctor}(undef, N_total_kinetic)

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
                            F_macro_vector_at_p = flux(system,U_macro_args...) # Splat into fluxes
                            
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
            if timestepper_name == "SimpleSplitting"
                #@assert isnothing(MainGrad) "SimpleSplitting only allowed for 1st order classic timestepper reference solution!"
                system_method_instance = SimpleSplitting(RalstonRK2(MainGrad, N_total; fallbackInterpolator = FallbackGrad, mood = mood_fun), relaxation_source, N_total)
            elseif timestepper_name == "ARS222" # Default IMEX choice
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

            @info "Starting 1D Euler system time integration..."
            elapsed_time, sys_xs_data, sys_us_data_kinetic, sys_ts_data = mainTimeIntegrator2!(
                system_method_instance, kinetic_eqs, component_pgs, sim_settings_obj
            )
            @info "1D Euler system time integration finished in $(round(elapsed_time, digits=2)) seconds."

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
        if initFunc_name == "eulerShockTube"
            calculateAllStats!(sim_data_result, (x,t) -> ic_object(x,t,system,base_particleGrid1D); discontinuity_points_func = t -> get_discontinuity_points(ic_object, system, t, base_particleGrid1D), quad_tol = 10e-9, dierckx_k = 4)
        end
    catch e
         if isa(e, KeyError); @error "Missing required parameter for System Simulation!" key=e.key params=run_params
         else; @error "Error during System Simulation!" params=run_params exception=(e, catch_backtrace()); end
         return nothing
    end
    return sim_data_result
end