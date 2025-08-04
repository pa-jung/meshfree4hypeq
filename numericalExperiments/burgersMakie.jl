# Place this code where your other simulation runner function is defined,
# ensuring access to necessary modules and types.

using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq.FluxFunctions
using Meshfree4ScalarEq.SourceTerms
using Meshfree4ScalarEq.ImplicitSolvers
#using Meshfree4ScalarEq.PlottingUtils
using Meshfree4ScalarEq.InitialConditions
using Random                    # For RNG state copy
using IPlotPDESols
using Meshfree4ScalarEq


"""

    runBurgersSimulation_for_IPlotPDESols(params::ParamDictType)

Runs a single 1D Burgers' equation simulation based on parameters defined
in the input dictionary and returns results compatible with IPlotPDESols.
Requires all necessary parameters to be present in the input `params` dict.
Calculates `dt` based on `CFL` and the Burgers equation properties.

# Arguments
- `params::ParamDictType`: Dictionary containing ALL necessary simulation parameters.
                           Expected keys: "method" (String matching names from burgers.txt),
                           "initFunc" (String), "N", "xmin", "xmax", "regular", "tmax",
                           "CFL", "saveFreq", "interpAlpha", "interpRangeFactor",
                           "randomness_factor" (if regular=false).

# Returns
- `AbstractSimData`: The simulation result object (e.g., SimData1D), or `nothing` on error.
"""
function RunSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    run_params = copy(params) # Work on a copy to store derived values

    # -----------------------------------------

    local sim_data_result = nothing # Ensure defined outside try

    try
        # --- Extract REQUIRED Parameters (Direct Access) ---
        # --- Extract REQUIRED Parameters (Direct Access) ---
        tmax::Float64 = run_params["tmax"]
        N_particles::Int = run_params["N"]
        bc::Symbol = run_params["bc"]
        xmin::Float64 = run_params["xmin"]
        xmax::Float64 = run_params["xmax"]
        initFunc_name::String = run_params["init_func"]
        cfl = get(run_params, "CFL", nothing)
        dt = get(run_params, "dt", nothing)
        snapshots::Int = run_params["snapshots"]

        eq_name = run_params["PDE"]

        # --- Extract METHOD Parameters (Use `get` with nothing defaults) ---
        order = get(run_params,"order",nothing)
        timestepper_name = get(run_params,"timestepper",nothing)
        interp_alpha = get(run_params,"interp_alpha",1.)
        interp_range_factor = get(run_params,"interp_range",1.)
        randomness_factor = get(run_params,"randomness_factor",0.)
        mood_name = get(run_params, "MOOD", nothing)
        delta_relax = get(run_params, "delta_relax", nothing)
        main_grad_name = get(run_params, "main_gradient", nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing) # Default fallback = 1st order Upwind
        main_flux_name = get(run_params, "main_flux", nothing)
        fallback_flux_name = get(run_params, "fallback_flux", nothing)
        switch_tol = get(run_params, "switch_tol", nothing)
        init_params = get(run_params, "init_params", nothing)
        relax_vel = get(run_params, "relax_velocities", nothing)
        relax_method = get(run_params, "relax_method", false)
        relax_eps = get(run_params, "relax_epsilon", nothing)
        eq_params = get(run_params, "PDE_params", nothing)
        lim = get(run_params, "limiter", nothing)
        seed_val = get(run_params, "SEED_value", nothing)

        if isnothing(order) || isnothing(timestepper_name); @info "Analytic solution detected!" end
        # --- Derive regularity ---
        regular::Bool = (randomness_factor == 0.0)
        N_ghost::Int64 = bc == :periodic ? 0 : convert(Int64, ceil(interp_range_factor))

        N = N_particles + 2*N_ghost
        rng = MersenneTwister(seed_val)

    
        @info "TimeStepper = $timestepper_name, Main Gradient = $main_grad_name ($order), Main Flux = $main_flux_name, 
                Fallback = $fallback_grad_name / $fallback_flux_name, MOOD = $mood_name (deltaRelax=$delta_relax), 
                IC = $initFunc_name, N = $N_particles, Regular = $regular (randFactor=$randomness_factor), Order = $order"

        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / N_particles
        local particleGrid

        randomness = randomness_factor * dx_nominal
        particleGrid = ParticleGrid1D(xmin, xmax, N_particles, N_ghost, bc;rng = rng, randomness = randomness)
        determineVolumes!(particleGrid)
        # --- Calculate Dependent Parameters ---
        interp_range = interp_range_factor * particleGrid.dx
        # Calculate dt based on CFL using the *correct* equation (Burgers)
        # --- Equation ---
        if eq_name == "burgers"
            eq = BurgersEquation() # Instantiate Burgers' equation object
        elseif eq_name == "linear"
            eq = LinearAdvection(eq_params...)
        else 
            error("Requested PDE not implemented!")
        end
        F = u -> flux(eq, u)
        # This assumes getTimeStep can handle BurgersEquation appropriately
        if !isnothing(cfl)
            eqLin = eq_name == "linear" ? eq : LinearAdvection(1.0)
            dt = cfl * getTimeStep(particleGrid, eqLin, interp_alpha, interp_range)
        elseif isnothing(dt)
            error("The time step has to be given directly via the dt-key or via the CFL fraction using the CFL-key!")
        end

        # --- NEW: Calculate save_frequency in steps ---
        if tmax <= 0 || dt <= 0
            # Handle edge case to avoid division by zero
            save_freq = 1 
        else
            # Calculate the desired time interval between saves
            save_time_interval = tmax / snapshots
            # Convert the time interval to an integer number of steps
            save_frequency_steps = round(Int, save_time_interval / dt)
            # Ensure we always take at least one step before saving
            save_freq = max(1, save_frequency_steps)
        end

        # --- Build Method Components (Ensure types/constructors are accessible) ---
        local mood_fun
        if mood_name == "U1"; mood_fun = MOODu1(deltaRelax = delta_relax)
        elseif mood_name == "U2"; mood_fun = MOODu2(deltaRelax = delta_relax)
        elseif mood_name == "LoubertU2"; mood_fun = MOODLoubertU2(deltaRelax = delta_relax)
        elseif mood_name == "none"; mood_fun = NoMOOD()
        elseif mood_name == "only"; mood_fun = OnlyMOOD()
        elseif mood_name == "firstNone"; mood_fun = FirstStageNoMOOD()
        elseif mood_name == "alt"; mood_fun = MOODAlt(delta_relax)
        elseif !isnothing(mood_name); error("MOOD '$mood_name' not recognized") end

        local limiter
        if !isnothing(lim)
            @assert (main_grad_name == "MUSCL") "Slope limiter only supported for MUSCL-schemes!"
            @assert (order == 2) "Only linear reconstruction supported at the moment!"
        end 
        if lim == "minmod"
            limiter = MinmodLimiter()
        elseif lim == "superbee"
            limiter = SuperbeeLimiter()
        elseif lim == "VK"
            limiter = VenkatakrishnanLimiter()
        elseif lim == "BJ"
            limiter = BarthJespersenLimiter()
        elseif !isnothing(lim); error("Limiter '$lim' not recognized") end
        local MainFlux
        if main_flux_name == "LW"; MainFlux = LaxWendroffFlux()
        elseif main_flux_name == "Rusanov"; MainFlux = RusanovFlux()
        elseif main_flux_name == "Upwind"; MainFlux = UpwindFlux()
        elseif !isnothing(main_flux_name); error("Requested Main Flux is not implemented!") end

        local FallbackFlux
        if fallback_flux_name == "LW"; FallbackFlux = LaxWendroffFlux()
        elseif fallback_flux_name == "Rusanov"; FallbackFlux = RusanovFlux()
        elseif fallback_flux_name == "Upwind"; FallbackFlux = UpwindFlux()
        elseif !isnothing(fallback_flux_name); error("Requested Fallback Flux is not implemented!") end

        local MainGrad
        if main_grad_name == "MUSCL"; MainGrad = isnothing(lim) ? MUSCL(order-1; numericalFlux = MainFlux) : MUSCLlimited(1; numericalFlux = MainFlux, limiter = limiter)
        #elseif main_grad_name == "MUSCLlimit"; MainGrad = MUSCLlimited(1; numericalFlux = MainFlux)
        elseif main_grad_name == "Upwind"; MainGrad = UpwindGradient(order; numericalFlux = MainFlux)
        elseif main_grad_name == "WENO"; error("WENO not implemented for non-linear case.")
        elseif !isnothing(main_grad_name); error("Requested Main GradientInterpolator not implemented!") end

        local FallbackGrad 
        if fallback_grad_name == "MUSCL"; error("No 1st order MUSCL method defined") #FallbackGrad = MUSCL(1; numericalFlux = FallbackFlux)
        elseif fallback_grad_name == "Upwind"; FallbackGrad = UpwindGradient(1; numericalFlux = FallbackFlux)
        elseif fallback_grad_name == "WENO"; error("WENO not implemented for non-linear case.")
        elseif isnothing(fallback_grad_name); FallbackGrad = nothing 
        else; error("Requested Fallback GradientInterpolator not implemented!") end
        # --- Time Stepper Selection ---
        local method
        if timestepper_name == "RalstonRK2"; method = RalstonRK2(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun)
        elseif timestepper_name == "EulerUpwind"; method = EulerUpwind(N; gradientInterpolator = MainGrad) # Assumes EulerUpwind ignores fallback/mood args if passed
        elseif timestepper_name == "RK3"; method = RK3(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun)
        elseif timestepper_name == "RK4"; method = RK4(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun)
        elseif timestepper_name == "LF"; method = LaxFriedrich(N)
        elseif timestepper_name == "LW"; method = ClassicalRichtmyerLWMOOD(N; mood = mood_fun)
        elseif timestepper_name == "Classic"; method = ClassicalTimeStepper(N, MainFlux)
        elseif timestepper_name == "Upwind"; method = Upwind(N)
        elseif timestepper_name == "RalstonRK2SmoothSwitch"; method = RalstonRK2SmoothSwitch2(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun, tol = switch_tol)
        elseif relax_method @info "Relaxation Method Detected!"
        elseif isnothing(timestepper_name); method = nothing 
        else; error("Unknown TimeStepper name: '$timestepper_name'"); end



        # --- Initial Condition ---
        local IC::InitialCondition
        IC = getInitialCondition(initFunc_name, init_params)
        analytic_func = (x,t) -> IC(x,t,eq, particleGrid)
        setInitialConditions!(particleGrid, x -> IC(x))
            # Create SimSettings object
        settings = SimSetting(  tmax=tmax,
                                dt=dt,
                                interpRange=interp_range,
                                interpAlpha= interp_alpha,
                                saveDir="/", 
                                saveFreq=save_freq,
                                organiseFiles = false)

        # --- Call the NEW Time Integrator ---
        if relax_method
            @assert !isnothing(relax_vel) "The relaxation method needs velocities to create the linear system!"
            eqs = [LinearAdvection(a) for a = relax_vel]
            pgs = [deepcopy(particleGrid) for _ = relax_vel]
            M = [(rho -> 1/2 * (rho + F(rho)/a)) for a = relax_vel]
            for (pg_idx,pg) = enumerate(pgs)
                for particle = pg.grid
                    particle.rho = M[pg_idx](particle.rho)
                end
            end
            source_term = RelaxationSourceTerm1D(M, relax_eps)
            implicit_solver = LinearizedRelaxationImplicitSolver()
            if timestepper_name =="ARS2"
                system_method = ARS2IMEX(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            elseif timestepper_name == "ARS233"
                system_method = ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            elseif timestepper_name == "PRSSP3"
                system_method = PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            elseif timestepper_name == "ARS222"
                system_method = ARS222(MainGrad,FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            elseif timestepper_name == "SSP2332"
                system_method = SSP2332(MainGrad,FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            else
                system_method = RelaxationStepper(method, N, M; epsilon = relax_eps)
            end
            elapsed_time, sys_xs, sys_us, ts = mainTimeIntegrator2!(system_method, eqs, pgs, settings)
            us = [vec(sum(sys_u, dims=2)) for sys_u = sys_us]
            xs = [sys_x[:,1] for sys_x = sys_xs]
        elseif !isnothing(method)
            elapsed_time, xs, us, ts = mainTimeIntegrator2!(method, eq, particleGrid, settings)
        else
            elapsed_time = 0.
            
            ts = collect(0:dt*save_freq:tmax)
            if ts[end] != tmax
                push!(ts, tmax)
            end
            xs = [collect(range(xmin, xmax, N_particles)) for _ = ts]
            us =  Vector{Vector{Float64}}(undef, 0)
            for (i,t) in enumerate(ts)
                u_tmp = map(x -> analytic_func(x,t), xs[i])
                push!(us, u_tmp)
            end
        end
        @info "Time integration finished in $(round(elapsed_time, digits=2)) seconds."

        sim_data_result = createSimData(xs, us, ts, run_params)
        # --- Post-processing ---
        if !isnothing(sim_data_result)
             # Add metadata to stats dictionary
             calculateAllStats!(sim_data_result, (x,t) -> IC(x,t,eq,particleGrid); discontinuity_points_func = t -> get_discontinuity_points(IC, eq, t, particleGrid), quad_tol = 10e-9, dierckx_k = 4)
             if hasproperty(sim_data_result, :stats) && isa(sim_data_result.stats, Dict)
                 sim_data_result.stats["time"] = elapsed_time
                #  sim_data_result.stats["dx_nominal"] = dx_nominal
                #  sim_data_result.stats["dt_calculated"] = dt
                #  sim_data_result.stats["interp_range_calculated"] = interp_range
                 #sim_data_result.stats["method_type"] = string(typeof(method))
             else
                 @warn "Could not add metadata to stats field in SimData."
             end
             # Ensure params field matches original input params for hashing consistency
             # If mainTimeIntegrator! modified run_params_for_integrator, copy original params back
             # sim_data_result.params = params # Check if necessary based on mainTimeIntegrator! behavior
        end
        return sim_data_result # Return the SimData object

    catch e
         if isa(e, KeyError); @error "Missing required parameter!" key=e.key params=params
         else; @error "Error during Burgers simulation setup or execution!" params=params exception=(e, catch_backtrace()); end
         return nothing # Return nothing on error
    end
end
# Example SimulationConfig for Burgers
sim_config_burgers = SimulationConfig(
    RunSimulation, # Use the new runner

    ParamDict(
        "tmax" => 5, "N" => 100, "xmin" => -5.0, "xmax" => 5.0,
        "CFL" => .2, "snapshots" => 5, "interp_alpha" => 1.0,
        "interp_range" => 3.5,
        "init_func" => "box",
        "init_params" => (0., 1., -2., 2.),#(0., 1., -4., -2.), #
        "randomness_factor" => 0.2, # Provide default needed when regular=false
        "SEED" => 10, 
        "bc" => :periodic,
        "order" => 1, "PDE" => "linear", "PDE_params" => (1.,)
    ),

    MethodDict(
        "RK2Upwind" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
        ),
        "RK2MUSCL2Smooth" => ParamDict(
            "timestepper" => "RalstonRK2SmoothSwitch",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "switch_tol" => 1e-8,
            "delta_relax" => 0.,
            "order" => 2
        ),
        "RK2MUSCL2(VKLimiter)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "limiter" => "VK",
            "MOOD" => "none",
        ),
        "RK2MUSCL2(Superbee)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "limiter" => "superbee",
            "MOOD" => "none",
        ),
        "RK2MUSCL2MOOD(U2)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(U2Relax)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 1.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(LoubertU2)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "LoubertU2",
            "delta_relax" => 1.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(U1)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U1",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(U1Relax)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U1",
            "delta_relax" => 5.,
            "order" => 2,
        ),
        "RK2MUSCL2" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 2
        ),
        "RK4MUSCL2" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 2,
        ),
        "RK4MUSCL5" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "MOOD" => "none",
        ),
        "RK4MUSCL5MOOD" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 5
        ),
        "EulerUpwind" => ParamDict(
            "timestepper" => "EulerUpwind",
            "main_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "order" => 1
        ),
        "Analytic Solution" => ParamDict(
            "N" => (:const, 1000), 
            "ignore" => ["interp_alpha", "randomness_factor", "SEED", "order"]
             # No randomness_factor needed when regular=true
        ),
        "LLF(uniform grid)" => ParamDict(
            "randomness_factor" => (:const,0.),
            "main_flux" => "Rusanov",
            "order" => 1,
            "timestepper" => "Classic",
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
                "ARS233MUSCL5" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "relax_method" => true,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none",
        ),
                "ARS233MUSCL5MOOD" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "relax_method" => true,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.,
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        # "PRSSP3MUSCL5" => ParamDict(
        #     "timestepper" => "PRSSP3",
        #     "main_gradient" => "MUSCL",
        #     "main_flux" => "Rusanov",
        #     "order" => 5,
        #     "relax_method" => true,
        #     "relax_velocities" => (1.,-1.),
        #     "relax_epsilon" => 10. ^ -8,
        #     "MOOD" => "none",
        # ),
        "ARS233MUSCL2" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "relax_method" => true,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none",
        ),
        "LWMOOD" => ParamDict(
            "timestepper" => "LW",
            "order" => 2,
            "MOOD" => "U1", "delta_relax" => 0.,
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
        "LW" => ParamDict(
            "timestepper" => "LW",
            "order" => 2,
            "MOOD" => "none",
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
        "ARS233MUSCL2MOOD" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "relax_method" => true,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.,
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        # "PRSSP3MUSCL2" => ParamDict(
        #     "timestepper" => "PRSSP3",
        #     "main_gradient" => "MUSCL",
        #     "main_flux" => "Rusanov",
        #     "MOOD" => "none",
        #     "order" => 2,
        #     "relax_method" => true,
        #     "relax_velocities" => (1.,-1.),
        #     "relax_epsilon" => 10. ^ -8,
        # )

    ),
    ["RK2MUSCL2","RK2MUSCL2MOOD(U2)", "ARS233MUSCL2MOOD"]#,"LW", "EulerUpwind"]
    #["LWMOOD","RK2MUSCL2", "LW","ARS233MUSCL2MOOD", "Analytic Solution", "RK2MUSCL2MOOD(U1)", "RK2MUSCL2MOOD(U2)", "RK4MUSCL5MOOD"]
    #["LWMOOD","RK2MUSCL2", "LW","ARS233MUSCL2MOOD", "Analytic Solution", "RK2MUSCL2MOOD(U1)", "RK2MUSCL2MOOD(U2)", "RK2MUSCL2MOOD(U2Relax)", "RK2MUSCL2MOOD(U1Relax)", "RK4MUSCL5MOOD"]#["RK2MUSCL2MOOD", "RK2MUSCL2", "RK4MUSCL5MOOD", "Analytic Solution"] #, "Relax Method 2", "Relax Method 3rd order","Classic","SlopeLimiter","SmoothSwitching","Regular MOOD", "OnlyFallback"]
    #["RK2MUSCL2Smooth", "Analytic Solution"]
    #["RK2MUSCL2MOOD(U1)", "Analytic Solution"]
);

# Pass this config to your IPlotPDESols functions
#show1DSolutionFig(sim_config_burgers; calc_stats = true, ui_options = :publication);
#showDynamicDependence(sim_config_burgers; ui_options = :publication)
#calculateConvergenceData(sim_config_burgers, "N", 10. .^(1:.25:2); force_int_param = true)
showConvergencePlot(sim_config_burgers, "N", 10. .^(1.5:.25:2); force_int_param = true, initial_calc = true, ui_options = :publication)
#showConvergencePlot(sim_config_burgers, "delta_relax", 10. .^(0.:0.05:1.5); force_int_param = false, initial_calc = false, ui_options = :publication)
#showConvergencePlot(sim_config_burgers, "switch_tol", 10. .^(-5:.1:-2); force_int_param = false, initial_calc = false, ui_options = :publication)
#showConvergencePlot(sim_config_burgers, "SEED", range(1,10000,5); force_int_param = true, initial_calc = true, ui_options = :publication);

