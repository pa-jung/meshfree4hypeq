# --- Module Imports ---
# Ensure all necessary modules from your project are accessible
using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.HyperbolicSystems
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
    runScalar2DSim(params::ParamDictType) -> Union{AbstractSimData, Nothing}

Runs a single 2D scalar advection simulation and returns results for IPlotPDESols.
This function is analogous to `runScalarSim` but adapted for 2D grids and equations.

# Arguments
- `params::ParamDictType`: A dictionary containing all simulation parameters.
                           Expected keys include "Nx", "Ny", "xmin", "xmax", "ymin", "ymax",
                           "PDE_params" (as a tuple for 2D velocity), etc.

# Returns
- `AbstractSimData`: A `SimData2D` object, or `nothing` if an error occurs.
"""
function runScalar2DSim(params::ParamDictType)::Union{AbstractSimData, Nothing}
    run_params = copy(params) # Work on a copy

    try
        # --- Extract REQUIRED 2D Parameters ---
        tmax::Float64 = run_params["tmax"]
        Nx::Int = run_params["Nx"]
        Ny::Int = run_params["Ny"]
        xmin::Float64 = run_params["xmin"]
        xmax::Float64 = run_params["xmax"]
        ymin::Float64 = run_params["ymin"]
        ymax::Float64 = run_params["ymax"]
        bc::Symbol = run_params["bc"]
        initFunc_name::String = run_params["init_func"]
        cfl = get(run_params, "CFL", nothing)
        dt = get(run_params, "dt", nothing)
        snapshots::Int = run_params["snapshots"]
        eq_name::String = run_params["PDE"]
        eq_params = get(run_params, "PDE_params", nothing)



        # --- Extract OPTIONAL Method Parameters ---
        order = get(run_params, "order", nothing)
        timestepper_name = get(run_params, "timestepper", nothing)
        interp_alpha = get(run_params, "interp_alpha", 1.0)
        interp_range_factor = get(run_params, "interp_range", 1.5)
        randomness_factor_tuple = get(run_params, "randomness_factor", (0.0, 0.0))
        mood_name = get(run_params, "MOOD", nothing)
        delta_relax = get(run_params, "delta_relax", nothing)
        main_grad_name = get(run_params, "main_gradient", nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing)
        main_flux_name = get(run_params, "main_flux", nothing)
        fallback_flux_name = get(run_params, "fallback_flux", nothing)
        init_params = get(run_params, "init_params", nothing)
        seed_val = get(run_params, "SEED_value", nothing)
        relax_vel = get(run_params, "relax_velocities", nothing)
        relax_eps = get(run_params, "relax_epsilon", nothing)
        save_relax = get(run_params, "save_relax", false)
        upwind_alg_2d = get(run_params, "upwind_alg_2d", "Classic") # Specific for 2D Upwind


        IC = getInitialCondition(initFunc_name, init_params)
        eq = LinearAdvection(eq_params)
        rng = MersenneTwister(seed_val)

        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / Nx
        dy_nominal = (ymax - ymin) / Ny
        N_ghost::Int = bc == :periodic ? 0 : get(run_params, "N_ghost", ceil(Int, interp_range_factor) + 1)
        Nx_total = Nx + 2*N_ghost
        Ny_total = Ny + 2*N_ghost
        randomness = (randomness_factor_tuple[1] * dx_nominal, randomness_factor_tuple[2] * dy_nominal)       
        particleGrid = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny, N_ghost, bc; rng=rng, randomness=randomness)
        determineVolumes!(particleGrid) # Essential for 2D to calculate Voronoi areas


        # --- Initial Condition ---
        setInitialConditions!(particleGrid, (x, y) -> IC(x, y))

        # --- ANALYTICAL SOLUTION BLOCK ---
        if isnothing(timestepper_name)
            @info "Calculating Analytical Solution for 2D Linear Advection..."

            
            # The ParticleGrid object is for passing domain info and bc type to the analytical solution
            #dummy_pg = ParticleGrid2D(xmin, xmax, ymin, ymax, 2, 2, 1, bc; randomness=randomness, rng = rng)
            analytic_func = (x, y, t) -> IC(x, y, t, eq, particleGrid)

            dt_snapshot = tmax > 0 ? tmax / snapshots : 0.0
            ts = tmax > 0 ? collect(0.0:dt_snapshot:tmax) : [0.0]
            if !isempty(ts) && abs(ts[end] - tmax) > 1e-9; push!(ts, tmax); end
            
            analytic_points = [(x,y) for y in range(ymin, ymax, length=Ny) for x in range(xmin, xmax, length=Nx)]
            xs = [analytic_points for _ in ts]
            us = [[analytic_func(p[1], p[2], t_snap) for p in analytic_points] for t_snap in ts]

            sim_data_result = createSimData(xs, us, ts, run_params)
            sim_data_result.stats["time"] = 0.0
            return sim_data_result
        end


        # --- Parameter Validation and Setup ---
        @assert eq_name == "linear" "Currently, only 2D Linear Advection is supported."
        @assert timestepper_name != "Analytical Solution" "Analytical solutions for 2D are not yet implemented in InitialConditions.jl."
        if get(run_params, "relax_method", false)
            @warn "2D relaxation systems are not yet supported. Ignoring relaxation parameters."
        end

        regular::Bool = (randomness_factor_tuple == (0.0, 0.0))

        @info "2D SIM: TimeStepper = $timestepper_name, Main Gradient = $main_grad_name ($order), N = ($Nx, $Ny), Regular = $regular"


        # --- Calculate Dependent Parameters ---
        interp_range = interp_range_factor * max(particleGrid.dx, particleGrid.dy)

        
        if !isnothing(cfl)
            dt = cfl * getTimeStep(particleGrid, eq, interp_alpha, interp_range)
        elseif isnothing(dt)
            error("A time step must be given via the 'dt' key or calculated via the 'CFL' key.")
        end

        save_freq = max(1, round(Int, (tmax / snapshots) / dt))


        settings = SimSetting(tmax, dt, interp_range, interp_alpha, save_freq)


        # --- Build Method Components ---
        mood_fun =   if mood_name == "U2"; MOODu2(deltaRelax=delta_relax)
                     elseif mood_name == "U1"; MOODu1(deltaRelax = delta_relax)
                     elseif mood_name == "none" || isnothing(mood_name); NoMOOD()
                     else error("MOOD '$mood_name' not recognized for 2D.")
                     end

        MainFlux = if main_flux_name == "Rusanov"; RusanovFlux()
                     elseif main_flux_name == "Upwind"; UpwindFlux()
                     else error("Main Flux '$main_flux_name' not implemented.")
                     end

        FallbackFlux = if fallback_flux_name == "Rusanov"; RusanovFlux()
                         elseif fallback_flux_name == "Upwind"; UpwindFlux()
                         elseif !isnothing(fallback_flux_name); error("Fallback Flux '$fallback_flux_name' not implemented.")
                         end
        
        MainGrad = if main_grad_name == "MUSCL"; MUSCL(order; numericalFlux=MainFlux)
                     elseif main_grad_name == "Upwind"; UpwindGradient(order; numericalFlux=MainFlux, algType=upwind_alg_2d)
                     elseif main_grad_name == "Central"; CentralGradient(order)
                     elseif main_grad_name == "WENO"; 
                     else error("Main Gradient '$main_grad_name' not implemented for 2D.")
                     end

        FallbackGrad = if isnothing(fallback_grad_name); nothing
                         elseif fallback_grad_name == "Upwind"; UpwindGradient(1; numericalFlux=FallbackFlux, algType=upwind_alg_2d)
                         elseif !isnothing(fallback_grad_name); error("Fallback Gradient '$fallback_grad_name' not implemented for 2D.")
                         end

        local source_term
        local implicit_solver
        local eqs
        local pgs
        if !isnothing(relax_vel)
            F = u -> flux(eq, u)[1]
            G = u -> flux(eq, u)[2]
            eqs = LinearAdvection{Tuple{Float64,Float64}}[]
            M = Function[]
            pgs = ParticleGrid2D[]
            for vel = relax_vel
                for a = [(vel,0.),(-vel,0.),(0.,vel),(0.,-vel)]
                    func = a[2] == 0. ? F : G
                    push!(eqs, LinearAdvection(a))
                    push!(pgs, deepcopy(particleGrid))
                    push!(M, rho -> 1/4 * (rho + 2 * func(rho)/sum(a)))
                end
            end
            for (pg_idx,pg) = enumerate(pgs)
                for particle = pg.grid
                    particle.rho = M[pg_idx](particle.rho)
                end
            end
            source_term = RelaxationSourceTerm(M, relax_eps, [1:length(M)])
            implicit_solver = LinearizedRelaxationImplicitSolver()            
        end
        # --- Time Stepper Selection ---
        method =    if timestepper_name == "RalstonRK2"; RalstonRK2(MainGrad, Nx_total, Ny_total; fallbackInterpolator=FallbackGrad, mood=mood_fun)
                    elseif timestepper_name == "RK4"; RK4(MainGrad, Nx_total, Ny_total; fallbackInterpolator=FallbackGrad, mood=mood_fun)
                    elseif timestepper_name == "RK3"; RK3(MainGrad, Nx_total, Ny_total; fallbackInterpolator=FallbackGrad, mood=mood_fun)
                    elseif timestepper_name == "Upwind"; method = Upwind(Nx_total, Ny_total)
                    elseif timestepper_name == "ARS233"; ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, Nx_total * Ny_total, 4*length(relax_vel))
                    elseif timestepper_name == "PRSSP3"; PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, Nx_total * Ny_total, 4*length(relax_vel))
                    elseif timestepper_name == "ARS222"; ARS222(MainGrad,FallbackGrad, mood_fun, implicit_solver, source_term, Nx_total * Ny_total, 4*length(relax_vel))
                    elseif timestepper_name == "IMEXRalstonRK2"; RalstonRK2(MainGrad,FallbackGrad, mood_fun, implicit_solver, source_term, Nx_total * Ny_total, 4*length(relax_vel))
                    elseif timestepper_name == "ARS232"; ARS232(MainGrad,FallbackGrad, mood_fun, implicit_solver, source_term, Nx_total * Ny_total, 4 * length(relax_vel))
                    elseif timestepper_name == "SimpleSplitting"; SimpleSplitting(RalstonRK2(MainGrad, Nx_total * Ny_total; fallbackInterpolator=FallbackGrad, mood=mood_fun), source_term, 4 * length(relax_vel))
                    else error("Unknown TimeStepper name: '$timestepper_name'")
                    end


        # --- Call Time Integrator ---
        if isnothing(relax_vel)
            elapsed_time, xs, us, ts = mainTimeIntegrator2!(method, eq, particleGrid, settings)
        else
            elapsed_time, sys_xs, sys_us, ts = mainTimeIntegrator2!(method, eqs, pgs, settings)
            us = save_relax ? sys_us : [vec(sum(sys_u, dims=2)) for sys_u = sys_us]
            xs = save_relax ? sys_xs : [sys_x[:,1] for sys_x = sys_xs]
        end            
        @info "2D Time integration finished in $(round(elapsed_time, digits=2)) seconds."
        
        local sim_data_result
        sim_data_result = createSimData(xs, us, ts, run_params)
        # --- Post-processing ---
        if !isnothing(sim_data_result) && hasproperty(sim_data_result, :stats)
            sim_data_result.stats["time"] = elapsed_time
            # Note: 2D error calculation (calculateAllStats!) would require a 2D analytical solution
            # and 2D integration, which is not yet implemented.
        end

        return sim_data_result

    catch e
        # if isa(e, KeyError)
        #     @error "Missing required 2D parameter!" key=e.key params=params
        # else
            @error "Error during 2D simulation!" params=params exception=(e, catch_backtrace())
#        end
        return nothing
    end
end
