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

"""
    runSystem2DSim(params::ParamDictType) -> Union{AbstractSimData, Nothing}

Runs a 2D system of conservation laws using a kinetic relaxation method.
This is the general-purpose runner for systems like 2D Euler, supporting
a flexible number of kinetic velocities per macroscopic variable.
"""
function runSystem2DSim(params::ParamDictType)::Union{AbstractSimData, Nothing}
    @info "\n--- Running 2D System Simulation (Relaxation Method) ---"
    run_params = copy(params)

    try
        # --- 1. Load All Required Parameters (Strict Loading) ---
        
        # Domain and Discretization
        tmax::Float64 = run_params["tmax"]
        Nx_interior::Int = run_params["Nx"]
        Ny_interior::Int = run_params["Ny"]
        xmin::Float64, xmax::Float64 = run_params["xmin"], run_params["xmax"]
        ymin::Float64, ymax::Float64 = run_params["ymin"], run_params["ymax"]
        bc::Symbol = run_params["bc"]
        
        # Physics and Initial Conditions
        system_name::String = run_params["PDE"]
        initFunc_name::String = run_params["init_func"]
        init_params = run_params["init_params"]
        
        # Relaxation Method
        relax_velocities_config::AbstractVector = run_params["relax_velocities"]
        relax_eps::Float64 = run_params["relax_epsilon"]

        # Numerical Scheme
        timestepper_name::String = run_params["timestepper"]
        main_grad_name::String = run_params["main_gradient"]
        fallback_grad_name = get(run_params,"fallback_gradient",nothing)
        order::Int = run_params["order"]
        main_flux_name::String = run_params["main_flux"]
        fallback_flux_name = get(run_params,"fallback_flux",nothing)
        mood_name::String = run_params["MOOD"]
        delta_relax = get(run_params,"delta_relax",nothing)
        
        # Time Stepping & Output
        cfl = get(run_params, "CFL", nothing)
        dt_val = get(run_params, "dt", nothing)
        snapshots::Int = run_params["snapshots"]
        
        # Meshfree Parameters
        interp_alpha::Float64 = run_params["interp_alpha"]
        interp_range_factor::Float64 = run_params["interp_range"]
        randomness_factor_tuple = run_params["randomness_factor"]
        upwind_alg_2d = get(run_params, "upwind_alg_2d", "Classic")
        seed_val = run_params["SEED_value"]

        @info "2D SYSTEM: $(system_name) with $timestepper_name, Main Grad: $main_grad_name (O$order), N=($Nx_interior, $Ny_interior)"

        # --- 2. Define Macroscopic System Physics ---
        local N_macro_vars::Int
        local system_eq::HyperbolicSystem
        local flux_funcs::Vector{Function}

        if system_name == "euler1d" # This runner can also do 1D systems if needed
            system_eq = Euler1D()
            N_macro_vars = 3 # rho, m, E
            flux_funcs = [ (U...) -> flux(system_eq, U...)[i] for i in 1:N_macro_vars ]
        elseif system_name == "euler2d"
             # For 2D, we assume a simple splitting of fluxes for this general model
            system_eq = Euler2D()
            N_macro_vars = 4 # rho, mx, my, E
            F_fluxes = [ (U...) -> flux(system_eq, U...)[1][i] for i in 1:N_macro_vars ]
            G_fluxes = [ (U...) -> flux(system_eq, U...)[2][i] for i in 1:N_macro_vars ]
        else
            error("System '$system_name' is not implemented.")
        end

        # --- 3. Construct the General Kinetic System ---
        num_kinetic_per_macro = [length(v) for v in relax_velocities_config]
        N_total_kinetic = sum(num_kinetic_per_macro)
        
        kinetic_eqs = Vector{LinearAdvection{Tuple{Float64,Float64}}}(undef, N_total_kinetic)
        M_funcs = Vector{Function}(undef, N_total_kinetic)
        
        kinetic_to_macro_map = Vector{Vector{Int}}(undef, N_macro_vars)
        current_offset = 0
        for i in 1:N_macro_vars
            kinetic_to_macro_map[i] = collect(current_offset+1 : current_offset+num_kinetic_per_macro[i])
            current_offset += num_kinetic_per_macro[i]
        end

        global_k_idx = 1
        for i_macro_loop in 1:N_macro_vars
            let i_macro = i_macro_loop # <--- CRITICAL FIX FOR SCOPING
                speeds_for_macro = relax_velocities_config[i_macro]
                # For 2D, flux is split. We assume a simple model where x-velocities link to F, y-velocities to G
                for speed_vec in speeds_for_macro
                    kinetic_eqs[global_k_idx] = LinearAdvection(speed_vec)
                    
                    # Determine which flux (F or G) to use based on velocity direction
                    local macro_flux_func
                    local relax_speed
                    if abs(speed_vec[1]) > 1e-12 # Has x-component
                        macro_flux_func = F_fluxes[i_macro]
                        relax_speed = speed_vec[1]
                    else # Has y-component
                        macro_flux_func = G_fluxes[i_macro]
                        relax_speed = speed_vec[2]
                    end

                    # This Maxwellian assumes a two-point quadrature model (+a, -a) for the chosen direction
                    M_funcs[global_k_idx] = (U...) -> 0.5 * (U[i_macro] + macro_flux_func(U...) / relax_speed)
                    
                    global_k_idx += 1
                end
            end
        end
        source_term = RelaxationSourceTerm(M_funcs, relax_eps, kinetic_to_macro_map)

        # --- 4. Grid & Initial Condition Setup ---
        N_ghost = bc == :periodic ? 0 : ceil(Int, interp_range_factor) + 1
        rng = MersenneTwister(seed_val)
        particleGrid_template = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx_interior, Ny_interior, N_ghost, bc; rng=rng, randomness=randomness_factor_tuple)
        
        IC = getInitialCondition(initFunc_name, init_params)
        macro_ic_at_points = [IC(p.pos...) for p in particleGrid_template.grid]

        particleGrids = [deepcopy(particleGrid_template) for _ in 1:N_total_kinetic]
        for k in 1:N_total_kinetic
            for p_idx in 1:length(particleGrids[k].grid)
                particleGrids[k].grid[p_idx].rho = M_funcs[k](macro_ic_at_points[p_idx]...)
            end
        end
        
        # --- 5. Time Step Calculation ---
        local dt::Float64
        if !isnothing(cfl)
            max_abs_speed = 0.0
            for speed_group in relax_velocities_config
                for speed_vec in speed_group
                    max_abs_speed = max(max_abs_speed, norm(speed_vec))
                end
            end
            if max_abs_speed < 1e-9; max_abs_speed = 1.0; end
            # Use a dummy 2D LA equation for getTimeStep
            temp_eq_for_dt = LinearAdvection((max_abs_speed, max_abs_speed))
            dt = cfl * getTimeStep(particleGrid_template, temp_eq_for_dt, interp_alpha, interp_range_factor * max(particleGrid_template.dx, particleGrid_template.dy))
        elseif !isnothing(dt_val)
            dt = dt_val
        else
            error("Either 'CFL' or 'dt' must be specified.")
        end
        
        save_freq = max(1, round(Int, (tmax / snapshots) / dt))
        settings = SimSetting(tmax=tmax, dt=dt, saveFreq=save_freq)
        
        # --- 6. Build Numerical Method ---
        mood_fun = if mood_name == "U2"; MOODu2(deltaRelax=delta_relax)
                   elseif mood_name == "U1"; MOODu1(deltaRelax=delta_relax)
                   else NoMOOD() end
        MainFlux = main_flux_name == "Upwind" ? UpwindFlux() : RusanovFlux()
        FallbackFlux = fallback_flux_name == "Upwind" ? UpwindFlux() : RusanovFlux()
        MainGrad = if main_grad_name == "MUSCL"; MUSCL(order; numericalFlux=MainFlux)
                   elseif main_grad_name == "Upwind"; UpwindGradient(order; numericalFlux=MainFlux, algType=upwind_alg_2d)
                   else error("Main Gradient '$main_grad_name' not supported.") end
        FallbackGrad = UpwindGradient(1; numericalFlux=FallbackFlux, algType=upwind_alg_2d)
        implicit_solver = LinearizedRelaxationImplicitSolver()
        N_total_particles = length(particleGrid_template.grid)
        
        system_method = if timestepper_name == "ARS233"; ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
                        elseif timestepper_name == "PRSSP3"; PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
                        elseif timestepper_name == "ARS222"; ARS222(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
                        elseif timestepper_name == "ARS232"; ARS232(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
                        elseif timestepper_name == "SimpleSplitting"; SimpleSplitting(RalstonRK2(MainGrad, N_total_particles; fallbackInterpolator=FallbackGrad, mood=mood_fun), source_term, N_total_particles)
                        else error("Unknown TimeStepper name for system: '$timestepper_name'") end

        # --- 7. Run Simulation ---
        elapsed_time, _, sys_us_kinetic, ts = mainTimeIntegrator2!(system_method, kinetic_eqs, particleGrids, settings)
        @info "2D System integration finished in $(round(elapsed_time, digits=2)) seconds."

        # --- 8. Post-process & Return ---
        interior_indices_vec = particleGrid_template.interior_indices
        us_macro = Vector{Matrix{Float64}}(undef, length(ts))
        for t_idx in eachindex(ts)
            us_macro[t_idx] = zeros(Float64, length(interior_indices_vec), N_macro_vars)
            for i_macro in 1:N_macro_vars
                for k_idx in kinetic_to_macro_map[i_macro]
                    kinetic_interior_data_k = [sys_us_kinetic[t_idx][p_idx, k_idx] for p_idx in interior_indices_vec]
                    us_macro[t_idx][:, i_macro] .+= kinetic_interior_data_k
                end
            end
        end
        interior_pos = [p.pos for p in particleGrid_template.grid[interior_indices_vec]]
        xs_final = [interior_pos for _ in ts]
        sim_data_result = createSimData(xs_final, us_macro, ts, run_params)
        sim_data_result.stats["time"] = elapsed_time
        return sim_data_result

    catch e
        @error "Error during 2D System simulation!" params=params exception=(e, catch_backtrace())
        return nothing
    end
end



# # --- Module Imports ---
# using Meshfree4ScalarEq.ScalarHyperbolicEquations
# using Meshfree4ScalarEq.HyperbolicSystems
# using Meshfree4ScalarEq.ParticleGrids
# using Meshfree4ScalarEq.TimeIntegration 
# using Meshfree4ScalarEq.Interpolations 
# using Meshfree4ScalarEq.SimSettings
# using Meshfree4ScalarEq.FluxFunctions
# using Meshfree4ScalarEq.SourceTerms 
# using Meshfree4ScalarEq.ImplicitSolvers 
# using Meshfree4ScalarEq.InitialConditions
# using Random
# using IPlotPDESols

# """
#     runSystem2DSim(params::ParamDictType) -> Union{AbstractSimData, Nothing}

# Runs a 2D system of conservation laws using a kinetic relaxation method.
# This is the general-purpose runner for systems like 2D Euler.
# """
# function runSystem2DSim(params::ParamDictType)::Union{AbstractSimData, Nothing}
#     @info "\n--- Running 2D System Simulation (Relaxation Method) ---"
#     run_params = copy(params)

#     try
#         # --- 1. Load All Required Parameters (Strict Loading) ---
        
#         # Domain and Discretization
#         tmax::Float64 = run_params["tmax"]
#         Nx_interior::Int = run_params["Nx"]
#         Ny_interior::Int = run_params["Ny"]
#         xmin::Float64, xmax::Float64 = run_params["xmin"], run_params["xmax"]
#         ymin::Float64, ymax::Float64 = run_params["ymin"], run_params["ymax"]
#         bc::Symbol = run_params["bc"]
        
#         # Physics and Initial Conditions
#         system_name::String = run_params["PDE"]
#         initFunc_name::String = run_params["init_func"]
#         init_params = run_params["init_params"]
        
#         # Relaxation Method
#         relax_a::Float64 = run_params["relax_velocities"]
#         relax_eps::Float64 = run_params["relax_epsilon"]

#         # Numerical Scheme
#         timestepper_name::String = run_params["timestepper"]
#         main_grad_name::String = run_params["main_gradient"]
#         fallback_grad_name::String = run_params["fallback_gradient"]
#         order::Int = run_params["order"]
#         main_flux_name::String = run_params["main_flux"]
#         fallback_flux_name::String = run_params["fallback_flux"]
#         mood_name::String = run_params["MOOD"]
#         delta_relax::Float64 = run_params["delta_relax"]
        
#         # Time Stepping & Output
#         cfl::Float64 = run_params["CFL"]
#         snapshots::Int = run_params["snapshots"]
        
#         # Meshfree Parameters
#         interp_alpha::Float64 = run_params["interp_alpha"]
#         interp_range_factor::Float64 = run_params["interp_range"]
#         randomness_factor_tuple = run_params["randomness_factor"]
#         upwind_alg_2d = get(run_params, "upwind_alg_2d", "Classic")
#         seed_val = run_params["SEED_value"]
        
#         @info "2D SYSTEM: $(system_name) with $timestepper_name, Main Grad: $main_grad_name (O$order), N=($Nx_interior, $Ny_interior)"

#         # --- 2. Define Macroscopic System Physics ---
#         local N_macro_vars::Int
#         local system_eq::HyperbolicSystem
#         local F_flux_funcs::Vector{Function}
#         local G_flux_funcs::Vector{Function}

#         if system_name == "euler2d"
#             system_eq = Euler2D()
#             N_macro_vars = 4 # rho, mx, my, E
#             F_flux_funcs = [ (U...) -> flux(system_eq, U...)[1][i] for i in 1:N_macro_vars ]
#             G_flux_funcs = [ (U...) -> flux(system_eq, U...)[2][i] for i in 1:N_macro_vars ]
#         else
#             error("System '$system_name' is not implemented.")
#         end

#         # --- 3. Construct the Kinetic System (4-velocity model per macro variable) ---
#         num_kinetic_per_macro = 4
#         N_total_kinetic = N_macro_vars * num_kinetic_per_macro
        
#         kinetic_eqs = Vector{LinearAdvection{Tuple{Float64,Float64}}}(undef, N_total_kinetic)
#         M_funcs = Vector{Function}(undef, N_total_kinetic)
#         kinetic_to_macro_map = [collect(1+(i-1)*num_kinetic_per_macro : i*num_kinetic_per_macro) for i in 1:N_macro_vars]

#         for i_macro in 1:N_macro_vars
#             F_i, G_i = F_flux_funcs[i_macro], G_flux_funcs[i_macro]
#             base_idx = (i_macro - 1) * num_kinetic_per_macro
            
#             kinetic_eqs[base_idx + 1] = LinearAdvection((relax_a, 0.0))
#             kinetic_eqs[base_idx + 2] = LinearAdvection((-relax_a, 0.0))
#             kinetic_eqs[base_idx + 3] = LinearAdvection((0.0, relax_a))
#             kinetic_eqs[base_idx + 4] = LinearAdvection((0.0, -relax_a))

#             M_funcs[base_idx + 1] = (U...) -> 0.25 * (U[i_macro] + 2 * F_i(U...) / relax_a)
#             M_funcs[base_idx + 2] = (U...) -> 0.25 * (U[i_macro] - 2 * F_i(U...) / relax_a)
#             M_funcs[base_idx + 3] = (U...) -> 0.25 * (U[i_macro] + 2 * G_i(U...) / relax_a)
#             M_funcs[base_idx + 4] = (U...) -> 0.25 * (U[i_macro] - 2 * G_i(U...) / relax_a)
#         end
#         source_term = RelaxationSourceTerm(M_funcs, relax_eps, kinetic_to_macro_map)

#         # --- 4. Grid & Initial Condition Setup ---
#         N_ghost = bc == :periodic ? 0 : ceil(Int, interp_range_factor) + 1
#         rng = MersenneTwister(seed_val)
#         particleGrid_template = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx_interior, Ny_interior, N_ghost, bc; rng=rng, randomness=randomness_factor_tuple)
        
#         IC = getInitialCondition(initFunc_name, init_params)
#         macro_ic_at_points = [IC(p.pos...) for p in particleGrid_template.grid]

#         particleGrids = [deepcopy(particleGrid_template) for _ in 1:N_total_kinetic]
#         for k in 1:N_total_kinetic
#             for p_idx in 1:length(particleGrids[k].grid)
#                 particleGrids[k].grid[p_idx].rho = M_funcs[k](macro_ic_at_points[p_idx]...)
#             end
#         end

#         # --- 5. Build Numerical Method ---
#         dt = cfl * min(particleGrid_template.dx, particleGrid_template.dy) / relax_a
#         save_freq = max(1, round(Int, (tmax / snapshots) / dt))
#         settings = SimSetting(tmax=tmax, dt=dt, saveFreq=save_freq)
        
#         mood_fun = if mood_name == "U2"; MOODu2(deltaRelax=delta_relax)
#                    elseif mood_name == "U1"; MOODu1(deltaRelax=delta_relax)
#                    elseif mood_name == "only"; OnlyMOOD()
#                    else NoMOOD()
#                    end
        
#         MainFlux = main_flux_name == "Upwind" ? UpwindFlux() : RusanovFlux()
#         FallbackFlux = fallback_flux_name == "Upwind" ? UpwindFlux() : RusanovFlux()
        
#         MainGrad = if main_grad_name == "MUSCL"; MUSCL(order; numericalFlux=MainFlux)
#                    elseif main_grad_name == "Upwind"; UpwindGradient(order; numericalFlux=MainFlux, algType=upwind_alg_2d)
#                    elseif main_grad_name == "Central"; CentralGradient(order)
#                    else error("Main Gradient '$main_grad_name' not supported.")
#                    end
        
#         FallbackGrad = UpwindGradient(1; numericalFlux=FallbackFlux, algType=upwind_alg_2d)
#         implicit_solver = LinearizedRelaxationImplicitSolver()
#         N_total_particles = length(particleGrid_template.grid)
        
#         system_method = if timestepper_name == "ARS233"; ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
#                         elseif timestepper_name == "PRSSP3"; PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
#                         elseif timestepper_name == "ARS222"; ARS222(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
#                         elseif timestepper_name == "ARS232"; ARS232(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
#                         elseif timestepper_name == "SimpleSplitting"; SimpleSplitting(RalstonRK2(MainGrad, N_total_particles; fallbackInterpolator=FallbackGrad, mood=mood_fun), source_term, N_total_particles)
#                         else error("Unknown TimeStepper name for system: '$timestepper_name'")
#                         end

#         # --- 6. Run Simulation ---
#         elapsed_time, _, sys_us_kinetic, ts = mainTimeIntegrator2!(system_method, kinetic_eqs, particleGrids, settings)
#         @info "2D System integration finished in $(round(elapsed_time, digits=2)) seconds."

#         # --- 7. Post-process: Recombine to Macroscopic Variables ---
#         interior_indices_vec = particleGrid_template.interior_indices
#         us_macro = Vector{Matrix{Float64}}(undef, length(ts))
#         for t_idx in eachindex(ts)
#             us_macro[t_idx] = zeros(Float64, length(interior_indices_vec), N_macro_vars)
#             for i_macro in 1:N_macro_vars
#                 for k_idx in kinetic_to_macro_map[i_macro]
#                     kinetic_interior_data_k = [sys_us_kinetic[t_idx][p_idx, k_idx] for p_idx in interior_indices_vec]
#                     us_macro[t_idx][:, i_macro] .+= kinetic_interior_data_k
#                 end
#             end
#         end
        
#         interior_pos = [p.pos for p in particleGrid_template.grid[interior_indices_vec]]
#         xs_final = [interior_pos for _ in ts]

#         sim_data_result = createSimData(xs_final, us_macro, ts, run_params)
#         sim_data_result.stats["time"] = elapsed_time

#         return sim_data_result

#     catch e
#         @error "Error during 2D System simulation!" params=params exception=(e, catch_backtrace())
#         return nothing
#     end
# end

