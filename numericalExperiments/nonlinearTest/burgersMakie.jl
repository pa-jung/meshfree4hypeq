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
function runBurgersSimulation_for_IPlotPDESols(params::ParamDictType)::Union{AbstractSimData, Nothing}
    println("\n--- Running Burgers Simulation via IPlotPDESols Interface ---")
    run_params = copy(params) # Work on a copy to store derived values

    # --- Manage RNG State for Reproducibility ---
    local rng_state_backup
    #rng_defined = @isdefined(Meshfree4ScalarEq.rng, :rng) && isa(Meshfree4ScalarEq.rng, Random.AbstractRNG)
    rng_state_backup = copy(Meshfree4ScalarEq.rng);
    # -----------------------------------------

    local sim_data_result = nothing # Ensure defined outside try

    try
        # --- Extract Parameters (Direct Access - throws KeyError if missing) ---
        tmax::Float64 = run_params["tmax"]
        N::Int = run_params["N"]
        xmin::Float64 = run_params["xmin"]
        xmax::Float64 = run_params["xmax"]
        initFunc_name::String = run_params["init_func"]
        method_name::String = run_params["method"] # Name like "muscl2RusanovFlux"
        cfl::Float64 = run_params["CFL"]
        interp_alpha::Float64 = run_params["interp_alpha"]
        save_freq::Int = run_params["save_frequency"] # For internal integrator steps
        interp_range_factor::Float64 = run_params["interp_range"]
        # Randomness factor required only if regular=false
        randomness_factor::Float64 = run_params["randomness_factor"]
        regular = run_params["randomness_factor"] == 0.0
        println("  Method Name = $method_name")
        println("  Initial Condition = $initFunc_name")
        println("  N = $N, Regular Grid = $regular")
        println("  tmax = $tmax, CFL = $cfl")
        println("----------------------------------------")

        # --- Method Selection (Based on Name String from burgers.txt) ---
        local method
        # Map method names from params["method"] to specific constructors
        if method_name == "muscl2RusanovFlux"
            method = RalstonRK2(MUSCL(2; numericalFlux=RusanovFlux()), N)
        elseif method_name == "lf"
             method = LaxFriedrich(N) # Assumes LaxFriedrich defined in TimeIntegration
        elseif method_name == "muscl2UpwindFlux"
             method = RalstonRK2(MUSCL(2; numericalFlux=UpwindFlux()), N)
        elseif method_name == "EulerUpwind"
            method = EulerUpwind(N)
        elseif method_name == "EulerMUSCL1"
            method = EulerUpwind(N; gradientInterpolator=MUSCL(1; numericalFlux=UpwindFlux()))
        elseif method_name == "muscl2RusanovFluxMoodu1"
            method = RalstonRK2(MUSCL(2; numericalFlux = RusanovFlux()), N; mood = MOODu1(deltaRelax = true))
        elseif method_name == "muscl2RusanovFluxMoodu2"
            method = RalstonRK2(MUSCL(2; numericalFlux = RusanovFlux()), N; mood = MOODu2(deltaRelax = true))
        elseif method_name == "muscl2RusanovFluxMoodu2RusanovFallback"
            method = RalstonRK2(MUSCL(2; numericalFlux = RusanovFlux()), N; mood = MOODu2(deltaRelax = true), fallbackInterpolator = UpwindGradient(1; algType = "Rusanov"))
        elseif method_name == "muscl2RusanovFluxMoodu2LFFallback"
            method = RalstonRK2(MUSCL(2; numericalFlux = RusanovFlux()), N; mood = MOODu2(deltaRelax = true), fallbackInterpolator = LaxFriedrichsGradient())
        elseif method_name == "MeshLFMoodu2"
            method = RalstonRK2(LaxFriedrichsGradient(), N; mood = MOODu2(deltaRelax = true), fallbackInterpolator = LaxFriedrichsGradient())         
        elseif method_name == "MeshLWMoodu2LFFallback"
            method = RalstonRK2(UpwindGradient(2; algType = "Rusanov"), N; mood = MOODu2(deltaRelax = true), fallbackInterpolator = LaxFriedrichsGradient())            
        # Add other methods from burgers.txt if needed...
        else
            error("Unknown method name provided in params for Burgers: '$method_name'")
        end
        run_params["method_type"] = string(typeof(method)) # Store descriptive name

        # --- Equation ---
        eq = BurgersEquation() # Instantiate Burgers' equation object
        run_params["equation"] = "BurgersEquation"

        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / N
        local particleGrid

        randomness = randomness_factor * dx_nominal
        particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = randomness)
        run_params["randomness_factor"] = randomness_factor # Record factor used

        # --- Calculate Dependent Parameters ---
        interp_range = interp_range_factor * particleGrid.dx
        # Calculate dt based on CFL using the *correct* equation (Burgers)
        # This assumes getTimeStep can handle BurgersEquation appropriately
        eqLin = LinearAdvection(1.0)
        dt = cfl * getTimeStep(particleGrid, eqLin, interp_alpha, interp_range)
        # Store actual values used
        run_params["interpRange"] = interp_range
        run_params["dt"] = dt

        # --- Initial Condition ---
        local init_func_handle::Function
        if initFunc_name == "smoothInit1"; init_func_handle = smoothInit1
        elseif initFunc_name == "smoothInit2"; init_func_handle = smoothInit2
        elseif initFunc_name == "shockInit1"; init_func_handle = shockInit1
        elseif initFunc_name == "shockInit2"; init_func_handle = shockInit2
        else; error("Unknown initFunc name: $initFunc_name"); end
        setInitialConditions!(particleGrid, init_func_handle)

        # --- Prepare final params for TimeIntegrator ---
        # Ensure keys match what mainTimeIntegrator! expects for its temp SimSetting
        run_params_for_integrator = copy(run_params) # Use the already updated run_params
        run_params_for_integrator["save_frequency"] = save_freq
        run_params_for_integrator["interp_range"] = interp_range
        run_params_for_integrator["interp_alpha"] = interp_alpha
        # Add tmax, dt explicitly if mainTimeIntegrator relies on them directly from dict
        run_params_for_integrator["tmax"] = tmax
        run_params_for_integrator["dt"] = dt


        # --- Call the NEW Time Integrator ---
        println("Starting time integration (Burgers)...")
        elapsed_time, sim_data_result = mainTimeIntegrator!(method, eq, particleGrid, run_params_for_integrator)
        println("Time integration finished in $(round(elapsed_time, digits=2)) seconds.")

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
    runBurgersSimulation_for_IPlotPDESols, # Use the new runner

    ParamDictType(
        "tmax" => 4.0, "N" => 100, "xmin" => -5.0, "xmax" => 5.0,
        "CFL" => 0.2, "save_frequency" => 2, "interp_alpha" => 1.0,
        "interp_range" => 3.5,
        "init_func" => "shockInit2",
        "randomness_factor" => 0.25, # Provide default needed when regular=false
        "SEED" => SEED_value
    ),

    MethodDictType(
        "muscl2RusanovFlux" => ParamDictType(
            # Uses shared params, no overrides needed here maybe?
        ),
        "lf" => ParamDictType(
            "randomness_factor" => (:const, 0.0)
             # No randomness_factor needed when regular=true
        ),
        "EulerMUSCL1" => ParamDictType(
             # Use different IC
        ),
        "muscl2UpwindFlux" => ParamDict(

        ),
        "EulerUpwind" => ParamDict(

        ),
        "muscl2RusanovFluxMoodu1" => ParamDict(
            
        ),
        "muscl2RusanovFluxMoodu2RusanovFallback" => ParamDict(

        ),
        "muscl2RusanovFluxMoodu2" => ParamDict(
            
        ),
        "muscl2RusanovFluxMoodu2LFFallback" => ParamDict(

        ),
        "MeshLFMoodu2" => ParamDict(

        ),
        "MeshLWMoodu2LFFallback" => ParamDict(

        )
    ),
    "lf"
)

# Pass this config to your IPlotPDESols functions
 show1DSolutionFig(sim_config_burgers)