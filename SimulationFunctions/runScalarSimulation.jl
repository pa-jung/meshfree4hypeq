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
using Meshfree4ScalarEq.MLSWeightFunctions
using Meshfree4ScalarEq.MOOD
using Random
using IPlotPDESols

"""
    runScalarSimulation(params::ParamDictType) -> Union{AbstractSimData, Nothing}

Runs a 1D or 2D scalar conservation law simulation.
This is the general-purpose runner for equations like Linear Advection and Burgers.
"""
function runScalarSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    @info "\n--- Running Scalar Simulation ---"
    run_params = copy(params)

    try
        # --- 1. Load Core Parameters ---
        tmax::Float64 = run_params["tmax"]
        xmin::Float64, xmax::Float64 = run_params["xmin"], run_params["xmax"]
        bc::Symbol = run_params["bc"]
        eq_name::String = run_params["PDE"]
        initFunc_name::String = run_params["init_func"]
        init_params = get(run_params, "init_params", nothing)
        timestepper_name = get(run_params, "timestepper", nothing)
        
        # --- 2. Determine Dimension and PDE Physics ---
        local dimension::Int
        local eq::ScalarHyperbolicPDE

        # Use the PDE name to determine dimension and create the equation object
        if eq_name == "linear"
            pde_params = run_params["PDE_params"]
            if pde_params isa Real
                dimension = 1
                eq = LinearAdvection(pde_params)
            else
                dimension = 2
                eq = LinearAdvection(Tuple(pde_params))
            end
        elseif eq_name == "burgers"
            dimension = 1
            eq = BurgersEquation()
        elseif eq_name == "burgers2d"
            dimension = 2
            eq = BurgersEquation2D()
        else
            error("Scalar PDE '$eq_name' is not implemented.")
        end

        IC = getInitialCondition(initFunc_name, init_params)
        snapshots::Int = run_params["snapshots"]
        
        # --- 3. REFACTORED: Handle Analytic Solution Case Early ---
        if isnothing(timestepper_name) || timestepper_name == "Analytic"
            @info "  Computing analytical solution for a D=$dimension scalar PDE..."
            
            # Setup a temporary grid to sample the solution
            local grid_analytic, xs
            ts = tmax > 0 ? collect(0.0:(tmax/snapshots):tmax) : [0.0]

            if dimension == 1
                Nx = run_params["N"]
                grid_analytic = ParticleGrid1D(xmin, xmax, Nx, bc != :periodic , bc)
                xs = grid_analytic.positions
                us = [[IC(x, t, eq, grid_analytic) for x in x_coords] for (x_coords, t) in zip(xs, ts)]
            else # dimension == 2
                Nx, Ny = run_params["Nx"], run_params["Ny"]
                ymin, ymax = run_params["ymin"], run_params["ymax"]
                grid_analytic = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny, bc != :periodic , bc, 0.)
                xs = grid_analytic.positions
                us = [[IC(p[1], p[2], t, eq, grid_analytic) for p in pos_coords] for (pos_coords, t) in zip(xs, ts)]

            end
            
            sim_data = createSimData(xs, us, ts, run_params)
            sim_data.stats["time"] = 0.0
            calculateAllStats!(sim_data, (x,t) -> IC(x,t,eq,grid_analytic); discontinuity_points_func = t -> get_discontinuity_points(IC, eq, t, grid_analytic), quad_tol = 10e-9, dierckx_k = 4)
            return sim_data
        end
        
        # --- 4. Load Remaining Numerical Parameters ---
        cfl = get(run_params, "CFL", nothing)
        dt = get(run_params, "dt", nothing)
        order = run_params["order"]
        interp_alpha = get(run_params, "interp_alpha", 1.0)
        interp_range_factor = get(run_params, "interp_range", 1.5)
        randomness_factor = get(run_params, "randomness_factor", 0.0)
        mood_name = get(run_params, "MOOD", nothing)
        delta_relax_factor = get(run_params, "delta_relax", 0)
        main_grad_name = get(run_params,"main_gradient",nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing)
        main_flux_name = get(run_params, "main_flux", nothing)
        fallback_flux_name = get(run_params, "fallback_flux", nothing)
        seed_val = get(run_params, "SEED_value", nothing)
        relax_vel = get(run_params, "relax_velocities", nothing)
        weight_func_name = get(run_params, "weight_function", nothing)
        lim = get(run_params, "limiter", nothing)

        @assert (isnothing(lim) || order == 2 || lim == "none") "Only 2nd order supported with limiter!"

        # --- 5. Grid Creation (Dimension-Aware) ---
        N_ghost::Int = bc == :periodic ? 0 : get(run_params, "N_ghost", ceil(Int, interp_range_factor) + 1)
        rng = MersenneTwister(seed_val)
        local particleGrid, interp_range, upwind_alg_2d

        if dimension == 1
            Nx = run_params["N"]
            dx_nominal = (xmax - xmin) / Nx
            randomness = randomness_factor * dx_nominal
            particleGrid = ParticleGrid1D(xmin, xmax, Nx, N_ghost, bc; rng=rng, randomness=randomness)
            interp_range = interp_range_factor * particleGrid.dx
            delta_relax = particleGrid.dx * delta_relax_factor
            upwind_alg_2d = "Classic"
        else # dimension == 2
            Nx, Ny = run_params["Nx"], run_params["Ny"]
            ymin, ymax = run_params["ymin"], run_params["ymax"]
            dx_nominal = (xmax - xmin) / Nx
            dy_nominal = (ymax - ymin) / Ny
            randomness = (randomness_factor[1] * dx_nominal, randomness_factor[2] * dy_nominal)
            particleGrid = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny, N_ghost, bc, interp_range_factor; rng=rng, randomness=randomness)
            interp_range = interp_range_factor * max(particleGrid.dx, particleGrid.dy)
            delta_relax = particleGrid.dx * particleGrid.dy * delta_relax_factor         
            upwind_alg_2d = main_grad_name == "Upwind" || fallback_grad_name == "Upwind" ? run_params["upwind_alg_2d"] : nothing
        end
        N_total_particles = particleGrid.N
        #determineVolumes!(particleGrid)
        setInitialConditions!(particleGrid, IC)

        weight_func = if weight_func_name == "exponential"; exponentialWeightFunction(interp_alpha, interp_range)
                      elseif !isnothing(weight_func_name) error("Weight function not implemented yet!") end
        updateNeighbors!(particleGrid, weight_func)
        # --- 6. Time Step and Settings ---
        if !isnothing(cfl)
            # For non-linear, use a dummy linear equation with max characteristic speed
            eq_for_dt = eq isa LinearAdvection ? eq : (dimension == 1 ? LinearAdvection(1.0) : LinearAdvection((1.,1.))) # Adjust max speed for Burgers if needed
            dt = cfl * getTimeStep(particleGrid, eq_for_dt)
        elseif isnothing(dt)
            error("Either 'dt' or 'CFL' must be provided.")
        end

        save_freq = max(1, round(Int, (tmax / snapshots) / dt))
        settings = SimSetting(tmax, dt, interp_range, interp_alpha, save_freq)

        limiter = if lim == "minmod"; MinmodLimiter()
                  elseif lim == "superbee"; SuperbeeLimiter()
                  elseif lim == "VK"; VenkatakrishnanLimiter()
                  elseif lim == "BJ"; BarthJespersenLimiter()
                  elseif lim == "none" || isnothing(lim); NoLimiter()
                  else error("Limiter '$lim' not recognized") end

        # --- Build Method Components ---
        mood_fun =   if mood_name == "U2"; MOODu2(deltaRelax=delta_relax)
                     elseif mood_name == "U1"; MOODu1(deltaRelax = delta_relax)
                     elseif mood_name == "only"; OnlyMOOD()
                     elseif mood_name == "none" || isnothing(mood_name); NoMOOD()
                     else error("MOOD '$mood_name' not recognized for 2D.")
                     end

        MainFlux = if main_flux_name == "Rusanov"; RusanovFlux()
                     elseif main_flux_name == "Upwind"; UpwindFlux()
                     elseif main_flux_name == "LW"; MainFlux = LaxWendroffFlux()
                     elseif !isnothing(main_flux_name); error("Main Flux '$main_flux_name' not implemented.")
                     end

        FallbackFlux = if fallback_flux_name == "Rusanov"; RusanovFlux()
                         elseif fallback_flux_name == "Upwind"; UpwindFlux()
                         elseif !isnothing(fallback_flux_name); error("Fallback Flux '$fallback_flux_name' not implemented.")
                         end
        
        is_classic = timestepper_name == "LW" || timestepper_name == "Classic" || timestepper_name == "LF"
        MainGrad = if main_grad_name == "MUSCL"; MUSCL(order-1, dimension; weightFunction = weight_func, numericalFlux = MainFlux, limiter = limiter)
                     elseif main_grad_name == "Upwind"; UpwindGradient(order, dimension; numericalFlux=MainFlux, algType=upwind_alg_2d, weightFunction=weight_func)
                     elseif main_grad_name == "Central"; CentralGradient(order, dimension; weightFunction=weight_func)
                     elseif main_grad_name == "WENO"; WENO(order, dimension; weightFunction = weight_func)
                     elseif main_grad_name == "DumbserWENO"; @assert dimension == 2 "DumbserWENO can only be used for 2D, for 1D use WENO instead!"; DumbserWENO(order; weightFunction = weight_func)
                     elseif !is_classic; error("Main Gradient '$main_grad_name' not implemented for 2D.")
                     end

        FallbackGrad = if isnothing(fallback_grad_name); NoFallbackGrad()
                         elseif fallback_grad_name == "Upwind"; UpwindGradient(1, dimension; numericalFlux=FallbackFlux, algType=upwind_alg_2d, weightFunction=weight_func)
                         elseif !isnothing(fallback_grad_name); error("Fallback Gradient '$fallback_grad_name' not implemented for 2D.")
                         end
        
        local xs, us, ts, elapsed_time, save_relax
        if isnothing(relax_vel)
            method = if timestepper_name == "RalstonRK2"; RalstonRK2(MainGrad, FallbackGrad, mood_fun)
            elseif timestepper_name == "EulerUpwind"; method = EulerUpwind(MainGrad) # Assumes EulerUpwind ignores fallback/mood args if passed
            elseif timestepper_name == "RK3"; method = RK3(MainGrad; fallbackInterpolator = FallbackGrad, mood = mood_fun)
            elseif timestepper_name == "RK4"; method = RK4(MainGrad; fallbackInterpolator = FallbackGrad, mood = mood_fun)
            elseif timestepper_name == "LF"; method = LaxFriedrich()
            elseif timestepper_name == "LW"; method = ClassicalRichtmyerLWMOOD(; mood = mood_fun)
            elseif timestepper_name == "Classic"; method = ClassicalTimeStepper(MainFlux)
            elseif timestepper_name == "Upwind"; method = Upwind(N_total_particles)
            elseif timestepper_name == "RalstonRK2SmoothSwitch"; method = RalstonRK2SmoothSwitch(MainGrad; fallbackInterpolator = FallbackGrad, mood = mood_fun, tol = run_params["switch_tol"])
            else error("Unknown Timestepper!") end

            save_relax = false
            # --- 8. Run Simulation ---
            elapsed_time, xs, us, ts = mainTimeIntegrator!(method, eq, particleGrid, settings; snapshots = snapshots)
        else
            relax_eps = run_params["relax_epsilon"]
            save_relax = run_params["save_relax"]
            N_kinetic = length(relax_vel)
            kinetic_eqs = ntuple(N_kinetic) do i; LinearAdvection(relax_vel[i]) end

            M_funcs_vec = Vector{MaxwellianFunctor}(undef, N_kinetic)
            coeff, int_factor = dimension == 1 ? (0.5, 1.0) : (0.25, 2.)
            for (i,speed) in enumerate(relax_vel)
                local i_dim::Int, relax_speed::Float64
                if dimension == 1
                    i_dim = 1
                    relax_speed = speed
                else # dimension == 2
                    i_dim = abs(speed[1]) > 1e-12 ? 1 : 2
                    relax_speed = speed[i_dim]
                end

                M_funcs_vec[i] = MaxwellianFunctor(eq, 1, i_dim, relax_speed, coeff, int_factor)
            end

            # BUG FIX 2: Correctly initialize the kinetic particle grids to be in equilibrium.
            pgs_vec = [deepcopy(particleGrid) for _ in 1:N_kinetic]
            for k in 1:N_kinetic
                for p_idx in 1:pgs_vec[k].N
                    # Get the macroscopic IC at this point
                    macro_ic_at_p = particleGrid.rhos[p_idx]
                    # Set the kinetic IC to be the Maxwellian evaluated at the macro IC
                    pgs_vec[k].rhos[p_idx] = M_funcs_vec[k]((macro_ic_at_p,))
                end
            end
            pgs = Tuple(pgs_vec)

            source_term = RelaxationSourceTerm(M_funcs_vec, relax_eps, [collect(1:N_kinetic)])
            implicit_solver = LinearizedRelaxationImplicitSolver()
            system_method = if timestepper_name == "ARS233"; ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term)
                            elseif timestepper_name == "PRSSP3"; PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term)
                            elseif timestepper_name == "ARS222"; ARS222(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term)
                            elseif timestepper_name == "ARS232"; ARS232(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term)
                            elseif timestepper_name == "SimpleSplitting"; SimpleSplitting(RalstonRK2(MainGrad; fallbackInterpolator=FallbackGrad, mood=mood_fun), source_term)
                            else error("Unknown TimeStepper name for system: '$timestepper_name'") 
                            end
            elapsed_time, sys_xs, sys_us, ts = mainTimeIntegrator!(system_method, kinetic_eqs, pgs, settings)
            @info "System integration (D=$dimension) finished in $(round(elapsed_time, digits=2)) seconds."

            us = save_relax ? sys_us : [vec(sum(sys_u, dims=2)) for sys_u = sys_us]
            xs = save_relax ? sys_xs : [sys_x[:,1] for sys_x = sys_xs]
        end
        @info "Scalar D=$dimension simulation finished in $(round(elapsed_time, digits=2)) seconds."

        sim_data_result = createSimData(xs, us, ts, run_params)
        if !save_relax
            if dimension == 1
                calculateAllStats!(sim_data_result, (x,t) -> IC(x,t,eq,particleGrid); discontinuity_points_func = t -> get_discontinuity_points(IC, eq, t, particleGrid), quad_tol = 10e-9, dierckx_k = 4)
            elseif dimension == 2
                #calculateAllStats!(sim_data_result, (x,t) -> IC(x[1],x[2],t,eq,particleGrid); discontinuity_points_func = t -> get_discontinuity_points(IC, eq, t, particleGrid), quad_tol = 10e-9, dierckx_k = 4)
            end
        end         
        
        sim_data_result.stats["time"] = elapsed_time
        return sim_data_result

    catch e
        @error "Error during Scalar simulation!" params=params exception=(e, catch_backtrace())
        return nothing
    end
end
