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
function shockInit1(x::Real) return x > 0.5 ? 1.0 : -1.0 end  # Adjusted from burgers.txt
function shockInit2(x::Real) return x > 0.5 ? 0 : 1.0 end  # Adjusted from burgers.txt
function shockInit3(x::Real) return ((x < -1.) | (x > 1.)) ? 0. : 1. end
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

        # --- Extract METHOD Parameters (Use `get` with sensible defaults) ---
        mood_name = get(run_params, "MOOD", nothing)
        delta_relax = get(run_params, "delta_relax", nothing)
        main_grad_name = get(run_params, "main_gradient", nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing) # Default fallback = 1st order Upwind
        main_flux_name = get(run_params, "main_flux", nothing)
        fallback_flux_name = get(run_params, "fallback_flux", nothing)

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
        elseif timestepper_name == "Upwind"; method = Upwind(N)
        else; error("Unknown TimeStepper name: '$timestepper_name'"); end

        # --- Equation ---
        eq = BurgersEquation() # Instantiate Burgers' equation object
        run_params["equation"] = "BurgersEquation"

        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / N
        local particleGrid

        randomness = randomness_factor * dx_nominal
        particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = randomness)

        # --- Calculate Dependent Parameters ---
        interp_range = interp_range_factor * particleGrid.dx
        # Calculate dt based on CFL using the *correct* equation (Burgers)
        # This assumes getTimeStep can handle BurgersEquation appropriately
        eqLin = LinearAdvection(1.0)
        dt = cfl * getTimeStep(particleGrid, eqLin, interp_alpha, interp_range)

        # --- Initial Condition ---
        local init_func_handle::Function
        if initFunc_name == "smoothInit1"; init_func_handle = smoothInit1
        elseif initFunc_name == "smoothInit2"; init_func_handle = smoothInit2
        elseif initFunc_name == "shockInit1"; init_func_handle = shockInit1
        elseif initFunc_name == "shockInit2"; init_func_handle = shockInit2
        elseif initFunc_name == "shockInit3"; init_func_handle = shockInit3
        else; error("Unknown initFunc name: $initFunc_name"); end
        setInitialConditions!(particleGrid, init_func_handle)

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
        elapsed_time, xs, us, ts = mainTimeIntegrator2!(method, eq, particleGrid, settings)
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
        "tmax" => 4.0, "N" => 100, "xmin" => -5.0, "xmax" => 5.0,
        "CFL" => 0.2, "save_frequency" => 2, "interp_alpha" => 1.0,
        "interp_range" => 3.5,
        "init_func" => "shockInit2",
        "randomness_factor" => 0.25, # Provide default needed when regular=false
        "SEED" => SEED_value, 
        "timestepper" => "LF",
        "order" => 1
    ),

    MethodDict(
        "Method 1" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Upwind",
            "MOOD" => "only",
            "delta_relax" => false,
            "order" => 2
        ),
        "Method 2" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Upwind",
            "MOOD" => "firstNone",
            "delta_relax" => false,
            "order" => 2
        ),
        "Method 3" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Upwind",
            "MOOD" => "none",
            "delta_relax" => false,
            "order" => 2
        ),
        "Method 4" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Upwind",
            "MOOD" => "U1",
            "delta_relax" => false,
            "order" => 2
        ),
        # "EulerUpwind" => ParamDict(
        #     "timestepper" => "EulerUpwind",
        #     "main_gradient" => "Upwind",
        #     "fallback_gradient" => "Upwind",
        #     "main_flux" => "Rusanov",
        #     "fallback_flux" => "Upwind",
        #     "MOOD" => "none",
        #     "delta_relax" => false,
        #     "order" => 1
        # ),
        "LF" => ParamDict(
            "randomness_factor" => (:const, 0.0)
             # No randomness_factor needed when regular=true
        )

    ),
    "Method 1";
    ui_options = Dict("animation_duration_s" => 10., "show_scatter" => true)
)

# Pass this config to your IPlotPDESols functions
 show1DSolutionFig(sim_config_burgers)