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
        upwind_alg_2d = get(run_params, "upwind_alg_2d", "Classic") # Specific for 2D Upwind

        # --- Parameter Validation and Setup ---
        @assert eq_name == "linear" "Currently, only 2D Linear Advection is supported."
        @assert timestepper_name != "Analytical Solution" "Analytical solutions for 2D are not yet implemented in InitialConditions.jl."
        if get(run_params, "relax_method", false)
            @warn "2D relaxation systems are not yet supported. Ignoring relaxation parameters."
        end

        regular::Bool = (randomness_factor_tuple == (0.0, 0.0))
        rng = MersenneTwister(seed_val)

        @info "2D SIM: TimeStepper = $timestepper_name, Main Gradient = $main_grad_name ($order), N = ($Nx, $Ny), Regular = $regular"

        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / Nx
        dy_nominal = (ymax - ymin) / Ny
        randomness = (randomness_factor_tuple[1] * dx_nominal, randomness_factor_tuple[2] * dy_nominal)
        println("Starting volume calc")        
        particleGrid = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny; rng=rng, randomness=randomness)
        println("particleGrid finished")
        determineVolumes!(particleGrid) # Essential for 2D to calculate Voronoi areas
        println("Finished volume calc")
        # --- Calculate Dependent Parameters ---
        interp_range = interp_range_factor * max(particleGrid.dx, particleGrid.dy)
        
        eq = LinearAdvection(eq_params)
        
        if !isnothing(cfl)
            dt = cfl * getTimeStep(particleGrid, eq, interp_alpha, interp_range)
        elseif isnothing(dt)
            error("A time step must be given via the 'dt' key or calculated via the 'CFL' key.")
        end

        save_freq = max(1, round(Int, (tmax / snapshots) / dt))

        # --- Build Method Components ---
        mood_fun = if mood_name == "U2"; MOODu2(deltaRelax=delta_relax)
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
                     else error("Main Gradient '$main_grad_name' not implemented for 2D.")
                     end

        FallbackGrad = if isnothing(fallback_grad_name); nothing
                         elseif fallback_grad_name == "Upwind"; UpwindGradient(1; numericalFlux=FallbackFlux, algType=upwind_alg_2d)
                         elseif !isnothing(fallback_grad_name); error("Fallback Gradient '$fallback_grad_name' not implemented for 2D.")
                         end

        # --- Time Stepper Selection ---
        method = if timestepper_name == "RalstonRK2"; RalstonRK2(MainGrad, Nx, Ny; fallbackInterpolator=FallbackGrad, mood=mood_fun)
                   elseif timestepper_name == "RK4"; RK4(MainGrad, Nx, Ny; fallbackInterpolator=FallbackGrad, mood=mood_fun)
                   else error("Unknown TimeStepper name: '$timestepper_name'")
                   end

        # --- Initial Condition ---
        IC = getInitialCondition(initFunc_name, init_params)
        # Note: 2D IC functor (x,y) must be implemented in InitialConditions.jl
        setInitialConditions!(particleGrid, (x, y) -> IC(x, y))

        settings = SimSetting(tmax=tmax, dt=dt, interpRange=interp_range, saveFreq = save_freq, interpAlpha=interp_alpha)

        # --- Call Time Integrator ---
        elapsed_time, xs, us, ts = mainTimeIntegrator2!(method, eq, particleGrid, settings)
        @info "2D Time integration finished in $(round(elapsed_time, digits=2)) seconds."
        
        sim_data_result = createSimData(xs, us, ts, run_params)

        # --- Post-processing ---
        if !isnothing(sim_data_result) && hasproperty(sim_data_result, :stats)
            sim_data_result.stats["time"] = elapsed_time
            # Note: 2D error calculation (calculateAllStats!) would require a 2D analytical solution
            # and 2D integration, which is not yet implemented.
        end

        return sim_data_result

    catch e
        if isa(e, KeyError)
            @error "Missing required 2D parameter!" key=e.key params=params
        else
            @error "Error during 2D simulation!" params=params exception=(e, catch_backtrace())
        end
        return nothing
    end
end
