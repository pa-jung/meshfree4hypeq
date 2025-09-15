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
using LinearAlgebra
using IPlotPDESols

"""
    runSystemSimulation(params::ParamDictType) -> Union{AbstractSimData, Nothing}

Runs a 1D or 2D system of conservation laws using a kinetic relaxation method.
This is the general-purpose runner for systems like Euler equations.
"""
function runSystemSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    @info "\n--- Running System Simulation (Relaxation Method) ---"
    run_params = copy(params)

    try
        # --- 1. Load Core Parameters ---
        tmax::Float64 = run_params["tmax"]
        xmin::Float64, xmax::Float64 = run_params["xmin"], run_params["xmax"]
        bc::Symbol = run_params["bc"]
        system_name::String = run_params["PDE"]
        initFunc_name::String = run_params["init_func"]
        init_params = run_params["init_params"]
        timestepper_name::String = run_params["timestepper"]
        snapshots::Int = run_params["snapshots"]
        
        # --- 2. Determine Dimension and System Physics ---
        local dimension::Int
        local N_macro_vars::Int
        local system_eq::HyperbolicPDESystem

        if system_name == "euler1d"
            dimension = 1
            system_eq = Euler1D()
            N_macro_vars = 3 # rho, m, E
        elseif system_name == "euler2d"
            dimension = 2
            system_eq = Euler2D()
            N_macro_vars = 4 # rho, mx, my, E
        else
            error("System '$system_name' is not implemented.")
        end
        
        # --- REFACTORED: Handle Analytic Solution Case Early ---
        IC = getInitialCondition(initFunc_name, init_params)
        if timestepper_name == "Analytic"
            @info "  Computing analytical solution..."
            # Setup a temporary grid to sample the solution
            grid_analytic = if dimension == 1
                ParticleGrid1D(xmin, xmax, run_params["N"], bc != :periodic , bc)
            else
                ParticleGrid2D(xmin, xmax, run_params["ymin"], run_params["ymax"], run_params["Nx"], run_params["Ny"], bc != :periodic, bc)
            end
            
            dt_analytic = tmax / snapshots
            ts = collect(0:dt_analytic:tmax)
            if ts[end] != tmax; push!(ts,tmax) end
            xs = [map(p -> p.pos, grid_analytic.grid) for _ in ts]
            us = Vector{Matrix{Float64}}(undef, length(ts))

            for (i, t) in enumerate(ts)
                us[i] = hcat([collect(IC(x, t, system_eq, grid_analytic)) for x in xs[i]]...)'
            end
            return createSimData(xs, us, ts, run_params)
        end
        
        # --- 3. Load Remaining Numerical Parameters ---
        relax_velocities_config::AbstractVector = run_params["relax_velocities"]
        relax_eps::Float64 = run_params["relax_epsilon"]
        main_grad_name::String = run_params["main_gradient"]
        fallback_grad_name = get(run_params,"fallback_gradient",nothing)
        order::Int = run_params["order"]
        main_flux_name = get(run_params,"main_flux",nothing)
        fallback_flux_name = get(run_params,"fallback_flux",nothing)
        lim = get(run_params, "limiter", nothing)
        mood_name::String = run_params["MOOD"]
        delta_relax = get(run_params,"delta_relax",nothing)
        cfl = get(run_params, "CFL", nothing)
        dt_val = get(run_params, "dt", nothing)
        interp_alpha::Float64 = run_params["interp_alpha"]
        interp_range_factor::Float64 = run_params["interp_range"]
        randomness_factor = run_params["randomness_factor"]
        seed_val = run_params["SEED"]
        weight_func_name = run_params["weight_function"]
        save_relax = run_params["save_relax"]
        
        # --- 4. Construct Kinetic System (Dimension-Aware) ---
        num_kinetic_per_macro = [length(v) for v in relax_velocities_config]
        N_total_kinetic = sum(num_kinetic_per_macro)
        
        kinetic_eqs_vec = Vector{LinearAdvection{dimension}}(undef, N_total_kinetic)
        M_funcs_vec = Vector{MaxwellianFunctor}(undef, N_total_kinetic)
        
        kinetic_to_macro_map = [
            collect(sum(num_kinetic_per_macro[1:i-1])+1 : sum(num_kinetic_per_macro[1:i]))
            for i in 1:N_macro_vars
        ]
        
        # Set Maxwellian parameters based on dimension
        coeff, int_factor = dimension == 1 ? (0.5, 1.0) : (0.25, 2.)

        global_k_idx = 1
        for i_macro in 1:N_macro_vars
            for speed in relax_velocities_config[i_macro]
                kinetic_eqs_vec[global_k_idx] = LinearAdvection(speed)
                
                local i_dim::Int, relax_speed::Float64
                if dimension == 1
                    i_dim = 1
                    relax_speed = speed
                else # dimension == 2
                    i_dim = abs(speed[1]) > 1e-12 ? 1 : 2
                    relax_speed = speed[i_dim]
                end

                M_funcs_vec[global_k_idx] = MaxwellianFunctor(system_eq, i_macro, i_dim, relax_speed, coeff, int_factor)
                global_k_idx += 1
            end
        end
        source_term = RelaxationSourceTerm(M_funcs_vec, relax_eps, kinetic_to_macro_map)
        
        # --- 5. Grid & Initial Condition Setup (Dimension-Aware) ---
        N_ghost = bc == :periodic ? 0 : ceil(Int, interp_range_factor) + 1
        rng = MersenneTwister(seed_val)
        
        local particleGrid_template, interp_range
        if dimension == 1
            Nx = run_params["N"]
            dx_nominal = (xmax - xmin) / Nx
            randomness = randomness_factor * dx_nominal
            particleGrid_template = ParticleGrid1D(xmin, xmax, Nx, N_ghost, bc; rng=rng, randomness=randomness)
            interp_range = interp_range_factor * particleGrid_template.dx
            macro_ic_at_points = [IC(p.pos) for p in particleGrid_template.grid]
        else # dimension == 2
            Nx, Ny = run_params["Nx"], run_params["Ny"]
            ymin, ymax = run_params["ymin"], run_params["ymax"]
            dx_nominal = (xmax - xmin) / Nx
            dy_nominal = (ymax - ymin) / Ny
            randomness = (randomness_factor[1] * dx_nominal, randomness_factor[2] * dy_nominal)
            particleGrid_template = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny, N_ghost, bc; rng=rng, randomness=randomness)
            interp_range = interp_range_factor * max(particleGrid_template.dx, particleGrid_template.dy)
            macro_ic_at_points = [IC(p.pos...) for p in particleGrid_template.grid]
        end
        
        # Initialize kinetic components to be in equilibrium with macroscopic IC
        particleGrids_vec = [deepcopy(particleGrid_template) for _ in 1:N_total_kinetic]
        for k in 1:N_total_kinetic, p_idx in 1:length(particleGrids_vec[k].grid)
            particleGrids_vec[k].grid[p_idx].rho = M_funcs_vec[k](macro_ic_at_points[p_idx])
        end
        
        # --- 6. Time Step Calculation (Dimension-Aware) ---
        local dt::Float64
        if !isnothing(cfl)
            max_abs_speed = maximum(norm(s) for group in relax_velocities_config for s in group)
            if max_abs_speed < 1e-9; max_abs_speed = 1.0; end
            
            temp_eq_for_dt = dimension == 1 ? LinearAdvection(max_abs_speed) : LinearAdvection((max_abs_speed, max_abs_speed))
            dt = cfl * getTimeStep(particleGrid_template, temp_eq_for_dt, interp_alpha, interp_range)
        else
            dt = dt_val
        end
        
        save_freq = max(1, round(Int, (tmax / snapshots) / dt))
        settings = SimSetting(tmax, dt, interp_range, interp_alpha, save_freq)
        
        # --- 7. Build Numerical Method & Run Simulation ---
        local mood_fun
        if mood_name == "U1"; mood_fun = MOODu1(deltaRelax = delta_relax)
        elseif mood_name == "U2"; mood_fun = MOODu2(deltaRelax = delta_relax)
        elseif mood_name == "LoubertU2"; mood_fun = MOODLoubertU2(deltaRelax = delta_relax)
        elseif mood_name == "none"; mood_fun = NoMOOD()
        elseif mood_name == "only"; mood_fun = OnlyMOOD()
        elseif !isnothing(mood_name); error("MOOD '$mood_name' not recognized") end

        local weight_func
        weight_func = if weight_func_name == "exponential"; exponentialWeightFunction()
                      else error("Weight function not implemented yet!") end

        local limiter
        if !isnothing(lim)
            @assert (main_grad_name == "MUSCL") "Slope limiter only supported for MUSCL-schemes!"
            @assert (order == 2) "Only linear reconstruction supported at the moment!"
        end 
        
        limiter = if lim == "minmod"; MinmodLimiter()
                  elseif lim == "superbee"; SuperbeeLimiter()
                  elseif lim == "VK"; VenkatakrishnanLimiter()
                  elseif lim == "BJ"; BarthJespersenLimiter()
                  elseif lim == "none"; NoLimiter()
                  elseif !isnothing(lim); error("Limiter '$lim' not recognized") end
                  

        MainFlux = if main_flux_name == "Rusanov"; RusanovFlux() 
            elseif main_grad_name!="WENO" error("Flux $main_flux_name NYI");
            elseif main_flux_name == "Upwind"; MainFlux = UpwindFlux() end
            
        FallbackFlux = if fallback_flux_name == "Rusanov"; RusanovFlux() 
                       elseif fallback_flux_name == "Upwind"; FallbackFlux = UpwindFlux()
                       elseif !isnothing(fallback_flux_name); error("Flux $fallback_flux_name NYI"); end
        upwind_alg_2d = dimension == 2 ? run_params["upwind_alg_2d"] : "Classic"
        MainGrad = if main_grad_name == "MUSCL"
                    isnothing(lim) ? MUSCL(order-1; weightFunction = weight_func, numericalFlux = MainFlux) : MUSCLlimited(1; weightFunction = weight_func, numericalFlux = MainFlux, limiter = limiter)
                    elseif main_grad_name == "WENO"
                        WENO(order; weightFunction = weight_func)
                    elseif main_grad_name == "MUSCLlimit"
                        MUSCLlimited(1; numericalFlux=MainFlux, weightFunction=weight_func)
                elseif main_grad_name == "Upwind"
                    UpwindGradient(1; numericalFlux=MainFlux, algType=upwind_alg_2d, weightFunction=weight_func)
                else error("Unknown MainGrad: $main_grad_name"); end
        FallbackGrad = if fallback_grad_name == "Upwind" UpwindGradient(1; numericalFlux=FallbackFlux, algType=upwind_alg_2d, weightFunction=weight_func)
                       elseif !isnothing(fallback_grad_name) error("Only Upwind implemented as Fallback!") end
        implicit_solver = LinearizedRelaxationImplicitSolver()
        N_total_particles = length(particleGrid_template.grid)
        
        system_method = if timestepper_name == "ARS233"; ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
                        elseif timestepper_name == "PRSSP3"; PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
                        elseif timestepper_name == "ARS222"; ARS222(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
                        elseif timestepper_name == "ARS232"; ARS232(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N_total_particles, N_total_kinetic)
                        elseif timestepper_name == "SimpleSplitting"; SimpleSplitting(RalstonRK2(MainGrad, N_total_particles; fallbackInterpolator=FallbackGrad, mood=mood_fun), source_term, N_total_particles)
                        else error("Unknown TimeStepper name for system: '$timestepper_name'") end
        
        # Convert to tuples for performance before passing to the integrator
        kinetic_eqs = Tuple(kinetic_eqs_vec)
        particleGrids = Tuple(particleGrids_vec)

        elapsed_time, sys_xs_data, sys_us_kinetic, ts = mainTimeIntegratorNew!(system_method, kinetic_eqs, particleGrids, settings)
        @info "System integration (D=$dimension) finished in $(round(elapsed_time, digits=2)) seconds."

        # --- 8. Post-process & Return ---
        
        local us_final, xs_final
        if save_relax
            us_final = sys_us_kinetic
            xs_final = sys_xs_data
        else
            us_final = map(sys_us_kinetic) do kinetic_data_at_t
                hcat([sum(eachcol(view(kinetic_data_at_t, :, indices))) for indices in kinetic_to_macro_map]...) 
            end
            xs_final = [sys_x[:,1] for sys_x = sys_xs_data]
        end

        sim_data_result = createSimData(xs_final, us_final, ts, run_params)
        sim_data_result.stats["time"] = elapsed_time
        return sim_data_result

    catch e
        @error "Error during System simulation!" params=params exception=(e, catch_backtrace())
        return nothing
    end
end
