# Place this code where your other simulation runner function is defined,
# ensuring access to necessary modules and types.

using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq.FluxFunctions
using Random                    # For RNG state copy
using IPlotPDESols
using Meshfree4ScalarEq
# --- Keep initial condition function definitions ---
function smoothInit1(x::Real) return exp(-x^2) end
function smoothInit2(x::Real) return sin(2*pi*x/5) + 1.0 end # Adjusted from burgers.txt


"""
    boxInit(x::Real, u_background::Real, u_box::Real, x_box_start::Real, x_box_end::Real)::Float64

Generates a box-like initial condition.
Returns `u_box` if `x_box_start <= x <= x_box_end`, and `u_background` otherwise.
"""
function boxInit(x::Real, u_background::Real, u_box::Real, x_box_start::Real, x_box_end::Real)::Float64
    if x_box_start <= x <= x_box_end
        return u_box
    else
        return u_background
    end
end

"""
    boxInitAna(x::Real, t::Real, u_background::Real, u_box::Real, x_box_start::Real, x_box_end::Real)::Float64

Provides the analytical solution for Burger's equation for a box initial condition,
assuming u_box > u_background.
The solution involves a rarefaction wave starting at x_box_start and a shock wave
starting at x_box_end.

# Arguments
- `x`: Spatial coordinate.
- `t`: Time.
- `u_background`: Value of u outside the box.
- `u_box`: Value of u inside the box.
- `x_box_start`: Left edge of the initial box.
- `x_box_end`: Right edge of the initial box.
"""
function boxInitAna(x::Real, t::Real, u_background::Real, u_box::Real, x_box_start::Real, x_box_end::Real)::Float64
    if t < 0.0
        error("Time t cannot be negative.")
    end

    if abs(u_box - u_background) < 1e-9 # Effectively constant state
        return u_background
    end
    
    if u_box < u_background
        error("This analytical solution is for u_box > u_background (bump/top-hat). For a trough, a different wave interaction occurs.")
    end

    if t == 0.0
        return boxInit(x, u_background, u_box, x_box_start, x_box_end)
    end

    # --- Phase 1 Characteristics (Waves evolve independently) ---

    # Rarefaction wave originating from x_box_start
    # Speeds in the fan range from u_background to u_box.
    x_rarefaction_tail = x_box_start + u_background * t
    x_rarefaction_head = x_box_start + u_box * t

    # Shock wave originating from x_box_end
    # Left state = u_box, Right state = u_background
    s_initial = (u_box + u_background) / 2.0
    x_shock_front_phase1 = x_box_end + s_initial * t

    # Time of interaction (when the u_box plateau vanishes)
    # Calculated from: x_rarefaction_head(t_int) = x_shock_front_phase1(t_int)
    # t_int * (u_box - s_initial) = x_box_end - x_box_start
    delta_u_main = u_box - u_background
    if abs(delta_u_main) < 1e-9 # Should have been caught by u_box approx u_background check
        return u_background # Avoid division by zero, effectively constant state
    end
    t_interaction = 2.0 * (x_box_end - x_box_start) / delta_u_main

    if t < t_interaction
        # --- Phase 1 Solution ---
        if x < x_rarefaction_tail
            return u_background
        elseif x < x_rarefaction_head 
            # Inside rarefaction fan: u(x,t) = (x - x_box_start)/t
            # This formula gives values in [u_background, u_box]
            # when x is between x_box_start + u_background*t and x_box_start + u_box*t
            return (x - x_box_start) / t
        elseif x < x_shock_front_phase1 
            # Plateau region
            return u_box
        else 
            # Behind shock
            return u_background
        end
    else # t >= t_interaction
        # --- Phase 2: Shock interacts with rarefaction ---
        # The shock path x_s(t) is:
        # x_s(t) = x_box_start + u_background*t + C*sqrt(t)
        # C = sqrt(2 * (x_box_end - x_box_start) * (u_box - u_background))
        
        C_factor = sqrt(2.0 * (x_box_end - x_box_start) * delta_u_main)
        x_shock_interacting = x_box_start + u_background * t + C_factor * sqrt(t)

        if x < x_rarefaction_tail 
            # Still to the left of the rarefaction's slowest part
            return u_background
        elseif x < x_shock_interacting 
            # Inside rarefaction fan, up to the interacting shock
            # u(x,t) = (x - x_box_start)/t
            return (x - x_box_start) / t
        else 
            # Behind the interacting shock
            return u_background
        end
    end
end

# Helper function to map a value back to the periodic domain [xmin, xmax)
function periodic_map(y::Real, xmin::Real, xmax::Real)
    L = xmax - xmin
    # Ensures the result is in [0, L) then shifts by xmin
    return xmin + mod(y - xmin, L)
end

# Function to numerically solve the implicit analytical solution for smooth ICs with periodicity
function solve_implicit_periodic(x::Real, t::Real, xmin::Real, xmax::Real, f::Function; tol::Real = 1e-9, max_iter::Int = 100)
    if t == 0.0
        return f(x)
    end

    u_old = f(x) # Initial guess

    for _ in 1:max_iter
        # Calculate the characteristic foot at t=0, mapped to the periodic domain
        xi = periodic_map(x - u_old * t, xmin, xmax)
        u_new = f(xi)
        if abs(u_new - u_old) < tol
            return u_new
        end
        u_old = u_new
    end
    # If iteration doesn't converge, return the last value and warn
    @warn "Fixed-point iteration for analytical solution did not converge at x=$x, t=$t. Max iterations reached."
    return u_old
end

# Analytical solution for smoothInit1 with periodicity (iterative evaluation)
function smoothInit1AnaPeriodic(x::Real, t::Real, xmin::Real, xmax::Real)
    return solve_implicit_periodic(x, t, xmin, xmax, smoothInit1)
end

# Analytical solution for smoothInit2 with periodicity (iterative evaluation)
function smoothInit2AnaPeriodic(x::Real, t::Real, xmin::Real, xmax::Real)
    return solve_implicit_periodic(x, t, xmin, xmax, smoothInit2)
end
# -------------------------------------------------

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
    println("\n--- Running Burgers Simulation via IPlotPDESols Interface ---")
    run_params = copy(params) # Work on a copy to store derived values

    # --- Manage RNG State for Reproducibility ---
    local rng_state_backup
    #rng_defined = @isdefined(Meshfree4ScalarEq.rng, :rng) && isa(Meshfree4ScalarEq.rng, Random.AbstractRNG)
    rng_state_backup = copy(Meshfree4ScalarEq.rng);
    # -----------------------------------------

    local sim_data_result = nothing # Ensure defined outside try

    try
        # --- Extract REQUIRED Parameters (Direct Access) ---
        # --- Extract REQUIRED Parameters (Direct Access) ---
        tmax::Float64 = run_params["tmax"]
        N::Int = run_params["N"]
        xmin::Float64 = run_params["xmin"]
        xmax::Float64 = run_params["xmax"]
        initFunc_name::String = run_params["init_func"]
        cfl::Float64 = run_params["CFL"]
        save_freq::Int = run_params["save_frequency"]
        interp_alpha::Float64 = run_params["interp_alpha"]
        interp_range_factor::Float64 = run_params["interp_range"]
        randomness_factor::Float64 = run_params["randomness_factor"] # Required
        order::Int = run_params["order"] # <<< Treat order as required
        timestepper_name = run_params["timestepper"] # Timestepper needed
        pde1_name = run_params["PDE1"]
        pde2_name = run_params["PDE2"]

        # --- Extract METHOD Parameters (Use `get` with sensible defaults) ---
        mood_name = get(run_params, "MOOD", nothing)
        delta_relax = get(run_params, "delta_relax", nothing)
        main_grad_name = get(run_params, "main_gradient", nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing) # Default fallback = 1st order Upwind
        main_flux_name = get(run_params, "main_flux", nothing)
        fallback_flux_name = get(run_params, "fallback_flux", nothing)
        switch_tol = get(run_params, "switch_tol", nothing)
        init_params = (get(run_params, "init_params", nothing))
        pde1_params = get(run_params, "PDE1_parameter", nothing)
        pde2_params = get(run_params, "PDE2_parameter", nothing)

        # --- Derive regularity ---
        regular::Bool = (randomness_factor == 0.0)

        println("  TimeStepper = $timestepper_name, Main Gradient = $main_grad_name ($order), Main Flux = $main_flux_name")
        println("  Fallback = $fallback_grad_name / $fallback_flux_name, MOOD = $mood_name (deltaRelax=$delta_relax)")
        println("  IC = $initFunc_name, N = $N, Regular = $regular (randFactor=$randomness_factor), Order = $order")
        println("  tmax = $tmax, CFL = $cfl")
        println("----------------------------------------")

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
        if main_grad_name == "MUSCL"; MainGrad = MUSCL(order-1; numericalFlux = MainFlux)
        elseif main_grad_name == "MUSCLlimit"; MainGrad = MUSCLlimited(1; numericalFlux = MainFlux)
        elseif main_grad_name == "Upwind"; MainGrad = UpwindGradient(order; numericalFlux = MainFlux)
        elseif main_grad_name == "WENO"; error("WENO not implemented for non-linear case.")
        elseif !isnothing(main_grad_name); error("Requested Main GradientInterpolator not implemented!") end

        local FallbackGrad 
        if fallback_grad_name == "MUSCL"; error("No 1st order MUSCL method defined") #FallbackGrad = MUSCL(1; numericalFlux = FallbackFlux)
        elseif fallback_grad_name == "Upwind"; FallbackGrad = UpwindGradient(1; numericalFlux = FallbackFlux)
        elseif fallback_grad_name == "WENO"; error("WENO not implemented for non-linear case.")
        elseif !isnothing(fallback_grad_name); error("Requested Fallback GradientInterpolator not implemented!") end
        println(run_params)
        # --- Time Stepper Selection ---
        local method
        if timestepper_name == "RalstonRK2"; method = RalstonRK2(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun)
        elseif timestepper_name == "EulerUpwind"; method = EulerUpwind(N; gradientInterpolator = MainGrad) # Assumes EulerUpwind ignores fallback/mood args if passed
        elseif timestepper_name == "RK3"; method = RK3(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun)
        elseif timestepper_name == "RK4"; method = RK4(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun)
        elseif timestepper_name == "LF"; method = LaxFriedrich(N)
        elseif timestepper_name == "Classic"; method = ClassicalTimeStepper(N, MainFlux)
        elseif timestepper_name == "Upwind"; method = Upwind(N)
        elseif timestepper_name == "Analytic"; method = nothing 
        elseif timestepper_name == "RalstonRK2Limiter"; method = RalstonRK2Limiter(MainGrad, N)
        elseif timestepper_name == "RalstonRK2SmoothSwitch"; method = RalstonRK2SmoothSwitch2(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun, tol = switch_tol)
        else; error("Unknown TimeStepper name: '$timestepper_name'"); end

        # --- Equation ---
        if pde1_name == "burgers"
            eq1 = BurgersEquation() # Instantiate Burgers' equation object
        elseif pde1_name == "linear"
            eq1 = LinearAdvection(pde1_params)
        else 
            error("Requested PDE not implemented!")
        end

        if pde2_name == "burgers"
            eq2 = BurgersEquation() # Instantiate Burgers' equation object
        elseif pde2_name == "linear"
            eq2 = LinearAdvection(pde2_params)
        else 
            error("Requested PDE not implemented!")
        end
        eqs = [eq1, eq2]

        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / N
        local particleGrids

        randomness = randomness_factor * dx_nominal
        particleGrids = [ParticleGrid1D(xmin, xmax, N; randomness = randomness) for i = 1:2]

        # --- Calculate Dependent Parameters ---
        interp_range = interp_range_factor * particleGrids[1].dx
        # Calculate dt based on CFL using the *correct* equation (Burgers)
        # This assumes getTimeStep can handle BurgersEquation appropriately
        eqLin = LinearAdvection(1.0)
        dt = cfl * getTimeStep(particleGrids[1], eqLin, interp_alpha, interp_range)

        # --- Initial Condition ---
        local init_func_handle::Function
        if initFunc_name == "smoothInit1"; init_func_handle = smoothInit1; analytic_func(x,t) = smoothInit1AnaPeriodic(x::Real, t::Real, xmin::Real, xmax::Real)
        elseif initFunc_name == "smoothInit2"; init_func_handle = smoothInit2
        elseif initFunc_name == "box"
            @assert typeof(init_params) <: Tuple{Real,Real,Real,Real} "Box Init needs a tuple of 4 real numbers as parameters!"
            u_background, u_box, box_start, box_end = init_params 
            @assert u_box >= u_background "Only top hat supported atm!"
            @assert box_end > box_start "The end of the box has to be larger than the start!"
            init_func_handle = x -> boxInit(x, u_background, u_box, box_start, box_end)
            analytic_func = (x,t) -> boxInitAna(x,t,u_background,u_box,box_start,box_end)
        else; error("Unknown initFunc name: $initFunc_name"); end
        setInitialConditions!(particleGrids, init_func_handle)
            # Create SimSettings object
        settings = SimSetting(  tmax=tmax,
                                dt=dt,
                                interpRange=interp_range,
                                interpAlpha= interp_alpha,
                                saveDir="/", 
                                saveFreq=save_freq,
                                organiseFiles = false)

        # --- Call the NEW Time Integrator ---
        println("Starting time integration (Burgers)...")
        if !isnothing(method)
            elapsed_time, xs, us, ts = mainTimeIntegrator2!(method, eqs, particleGrids, settings)
        else
            elapsed_time = 0.
            dx = (xmax - xmin) * 10^-3
            dt = (tmax - 0) * 10^-2
            ts = collect(0:dt:tmax)
            xs = [collect(xmin:dx:xmax) for _ = eachindex(ts)]
            us =  Vector{Vector{Float64}}(undef, 0)
            for (i,t) in enumerate(ts)
                u_tmp = map(x -> analytic_func(x,t), xs[i])
                push!(us, u_tmp)
            end
        end
        println(typeof(xs), typeof(us), typeof(ts))
        println("Time integration finished in $(round(elapsed_time, digits=2)) seconds.")

        sim_data_result = createSimData(xs, us, ts, run_params)
        # --- Post-processing ---
        if !isnothing(sim_data_result)
             # Add metadata to stats dictionary
             if hasproperty(sim_data_result, :stats) && isa(sim_data_result.stats, Dict)
                 sim_data_result.stats["time"] = elapsed_time
                 sim_data_result.stats["dx_nominal"] = dx_nominal
                 sim_data_result.stats["dt_calculated"] = dt
                 sim_data_result.stats["interp_range_calculated"] = interp_range
                 sim_data_result.stats["method_type"] = string(typeof(method))
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
    finally
        # Restore RNG state regardless of success/failure
        copy!(Meshfree4ScalarEq.rng, rng_state_backup)
    end
end
SEED_value = (:const, Meshfree4ScalarEq.SEED)
# Example SimulationConfig for Burgers
sim_config_burgers = SimulationConfig(
    RunSimulation, # Use the new runner

    ParamDict(
        "tmax" => 5.0, "N" => 100, "xmin" => -5.0, "xmax" => 5.0,
        "CFL" => 0.2, "save_frequency" => 2, "interp_alpha" => 1.0,
        "interp_range" => 3.5,
        "init_func" => "box",
        "init_params" => (0., 1., -5.,0.),
        "PDE1" => "linear",
        "PDE2" => "linear",
        "PDE1_parameter" => 1.,
        "PDE2_parameter" => .5,
        "randomness_factor" => 0.25, # Provide default needed when regular=false
        "SEED" => SEED_value, 
        "timestepper" => "Classic",
        "order" => 1, "PDE" => "burgers"
    ),

    MethodDict(
        "OnlyFallback" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "only",
            "delta_relax" => false,
            "order" => 2
        ),
        "SmoothSwitching" => ParamDict(
            "timestepper" => "RalstonRK2SmoothSwitch",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U1",
            "switch_tol" => .0025,
            "delta_relax" => true,
            "order" => 2
        ),
        "SlopeLimiter" => ParamDict(
            "timestepper" => "RalstonRK2Limiter",
            "main_gradient" => "MUSCLlimit",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "none",
            "delta_relax" => false,
            "order" => 2
        ),
        "Regular MOOD" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U1",
            "delta_relax" => true,
            "order" => 2
        ),
            "No MOOD" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "none",
            "delta_relax" => false,
            "order" => 2
        ),
        "EulerUpwind" => ParamDict(
            "timestepper" => "EulerUpwind",
            "main_gradient" => "Upwind",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "none",
            "delta_relax" => false,
            "order" => 1
        ),
        "Analytic" => ParamDict(
            "timestepper" => "Analytic",
            "randomness_factor" => (:const,0.)
             # No randomness_factor needed when regular=true
        ),
        "Classic" => ParamDict(
            "randomness_factor" => (:const,0.),
            "main_flux" => "Rusanov"
        )

    ),
    ["Classic"];
    ui_options = Dict("animation_duration_s" => 10., "show_scatter" => false, "system_dimension" => 2)
)

# Pass this config to your IPlotPDESols functions
show1DSolutionFig(sim_config_burgers);