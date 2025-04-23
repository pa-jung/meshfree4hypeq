using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq

using IPlotPDESols

# --- Keep initial condition function definitions ---
function smoothInit1(x::Real) return exp(-x^2) end
function smoothInit2(x::Real) return sin(2*pi*x/5) end
function shockInit(x::Real) return x > 0.0 ? 1.0 : 0.0 end
# -------------------------------------------------

# --- Keep TimeStepper definitions and includes ---
# abstract type TimeStepper end ... etc.
# include("FixedGridTimeSteppers.jl") ... etc.
# --- Keep definition of the NEW mainTimeIntegrator! ---
# function mainTimeIntegrator!(timeStepper::TimeStepper, eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, params::ParamDictType) ... end

"""
    runSimulation_for_IPlotPDESols(params::ParamDictType)

Runs a single 1D simulation based on parameters defined in the input dictionary
and returns the results in an AbstractSimData object. Requires all necessary
parameters to be explicitly present in the input `params` dict (via merging
shared and method-specific parameters from SimulationConfig). Calculates `dt` based on `CFL`.

# Arguments
- `params::ParamDictType`: Dictionary containing ALL necessary simulation parameters.
                           Expected keys: "method" (String), "initFunc" (String),
                           "N", "xmin", "xmax", "regular", "tmax", "CFL", "saveFreq",
                           "interpAlpha", "interpRangeFactor", "advection_velocity" (optional).

# Returns
- `AbstractSimData`: The simulation result object (e.g., SimData1D), or `nothing` on error.
"""
function runSimulation_for_IPlotPDESols(params::ParamDictType)::Union{AbstractSimData, Nothing}
    println("\n--- Running Simulation via IPlotPDESols Interface ---")
    run_params = copy(params) # Work on a copy to store derived values

    state = copy(Meshfree4ScalarEq.rng)

    try
        # --- Extract Parameters (Direct Access - throws KeyError if missing) ---
        # These MUST be provided in the `params` dict passed to this function
        # (typically by merging shared_params and methods_dict[method_key] in the plotting func)
        tmax::Float64 = run_params["tmax"]
        N::Int = run_params["N"]
        xmin::Float64 = run_params["xmin"]
        xmax::Float64 = run_params["xmax"]
        regular::Bool = run_params["regular"]
        initFunc_name::String = run_params["initFunc"]
        method_name::String = run_params["method"] # Expect the name used for selection
        cfl::Float64 = run_params["CFL"]
        interp_alpha::Float64 = run_params["interpAlpha"]
        save_freq::Int = run_params["saveFreq"] # For internal integrator steps
        interp_range_factor::Float64 = run_params["interpRangeFactor"]
        a::Float64 = get(run_params, "advection_velocity", 1.0) # Optional: Keep get only for truly optional params

        # Print essential params for debugging/tracking
        println("  Method Name = $method_name")
        println("  Initial Condition = $initFunc_name")
        println("  N = $N, Regular Grid = $regular")
        println("  tmax = $tmax, CFL = $cfl")
        # ... add others if needed ...
        println("----------------------------------------")


        # --- Method Selection (Based on Name String) ---
        #local method::TimeStepper
        # Map method names from params["method"] to constructors
        if method_name == "Upwind"; method = Upwind(N)
        elseif method_name == "EulerUpwind"; method = EulerUpwind(N)
        elseif method_name == "RK2MUSCL1"; method = RalstonRK2(MUSCL(1), N)
        elseif method_name == "RK3WENO2"; method = RK3(WENO(2), N)
        # Add logic for MOOD types - ensure they are defined/accessible
        elseif method_name == "RK3MOODMUSCL1"; method = RK3(MUSCL(1), N; mood = MOODu1(deltaRelax=true))
        elseif method_name == "RK3MOODMUSCL2"; method = RK3(MUSCL(2), N; mood = MOODu2(deltaRelax=true))
        # Add mappings for all method names you intend to use...
        # Example: Ensure names here match keys/values in SimulationConfig.methods_dict
        elseif method_name == "RK3MUSCL3"; method = RK3(MUSCL(3), N)
        elseif method_name == "RK4MUSCL4NoMOOD"; method = RK3(MUSCL(4), N; mood = NoMOOD())
        elseif method_name == "RK4MOODu1MUSCL4"; method = RK3(MUSCL(4), N; mood = MOODu1(deltaRelax=true))
        elseif method_name == "RK4MOODu2MUSCL4"; method = RK3(MUSCL(4), N; mood = MOODu2(deltaRelax=true))
        else; error("Unknown method name provided in params: '$method_name'"); end
        run_params["method_type"] = string(typeof(method)) # Store resolved type name

        # --- Equation ---
        eq = LinearAdvection(a)
        run_params["equation"] = "LinearAdvection(a=$a)"
        run_params["advection_velocity"] = a # Store 'a' used

        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / N
        local particleGrid::ParticleGrid1D
        if regular
            particleGrid = ParticleGrid1D(xmin, xmax, N)
        else
            # Require randomness factor if irregular
            randomness_factor = run_params["randomness_factor"]::Float64 # Assume required if irregular=false
            randomness = randomness_factor * dx_nominal
            particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = randomness)
        end
        run_params["dx_nominal"] = dx_nominal # Store nominal dx

        # --- Calculate Dependent Parameters ---
        interp_range = interp_range_factor * particleGrid.dx # Use grid's actual dx
        # Calculate dt based on CFL - `dt` key in input params is ignored now
        dt = cfl * getTimeStep(particleGrid, eq, interp_alpha, interp_range) # Assume getTimeStep exists
        # Store actual values used back into the dictionary for saving
        run_params["interpRange"] = interp_range
        run_params["dt"] = dt # Store the calculated dt
        # run_params["CFL"] = cfl # CFL is already in run_params

        # --- Initial Condition ---
        local init_func_handle::Function
        if initFunc_name == "smoothInit1"; init_func_handle = smoothInit1
        elseif initFunc_name == "smoothInit2"; init_func_handle = smoothInit2
        elseif initFunc_name == "shockInit"; init_func_handle = shockInit
        else; error("Unknown initFunc name: $initFunc_name"); end
        setInitialConditions!(particleGrid, init_func_handle)
        # run_params["initFunc"] already contains the name

        # --- Add SimSetting equivalents needed by mainTimeIntegrator! ---
        # Ensure these keys match exactly what mainTimeIntegrator! expects internally
        run_params["save_frequency"] = save_freq
        run_params["interp_range"] = interp_range # Pass calculated range
        run_params["interp_alpha"] = interp_alpha # Pass alpha

        # --- Call the NEW Time Integrator ---
        println("Starting time integration...")
        # This version expects ParamDictType and returns (elapsed_time, sim_data)
        elapsed_time, sim_data_result = mainTimeIntegrator!(method, eq, particleGrid, run_params)
        println("Time integration finished in $(round(elapsed_time, digits=2)) seconds.")

        # Add elapsed time to stats? (Optional)
        if !isnothing(sim_data_result) && hasproperty(sim_data_result, :stats) && isa(sim_data_result.stats, Dict)
             sim_data_result.stats["time"] = elapsed_time
        end
        copy!(Meshfree4ScalarEq.rng, state)

        # The sim_function in SimulationConfig should return the SimData object
        return sim_data_result

    catch e
         # Catch specific KeyError for missing parameters
         if isa(e, KeyError)
             @error "Missing required parameter in input dictionary!" key=e.key params=params
         else
              # Catch other errors during setup or execution
              @error "Error during simulation setup or execution!" params=params exception=(e, catch_backtrace())
         end
         copy!(Meshfree4ScalarEq.rng, state)
         return nothing # Return nothing to indicate failure
    end
end

# --- Example SimulationConfig Usage ---

shared_params =     ParamDict( # Define ALL common defaults needed by runner
"tmax" => 1.0, "N" => 50, "xmin" => -5.0, "xmax" => 5.0,
"CFL" => 0.4, "saveFreq" => 5, "interpAlpha" => 6.0,
"interpRangeFactor" => 3.5, "advection_velocity" => 1.0,
"regular" => true, 
"initFunc" => "smoothInit1", #Ensure ALL keys expected by runSimulation... are here or overridden below
# If regular=false requires randomness_factor, provide a default here if desired
"randomness_factor" => 0.5 # Example default
)

SEED_value = (:const, Meshfree4ScalarEq.SEED)
methods_dict =     MethodDict(
    # Keys are the method name strings used in the if/elseif block of the runner
    # Values dictionary contains ONLY parameters DIFFERENT from shared_params
    # OR parameters specific only to this method type.
   "EulerUpwind" => ParamDict( # Key is the actual method name
       # Overrides shared_params:
       "regular" => false,
       "SEED" => SEED_value
       # Specific param needed because regular=false (assuming no default in shared):
   ),
   "RK3WENO2" => ParamDict(
       # Overrides shared_params:
       "regular" => false,
       "SEED" => SEED_value
       # Inherits regular=true, N, tmax, CFL etc. from shared_params
   ),
    "Upwind" => ParamDict(
       # Inherits all defaults from shared_params
    )
   # Add other methods, using their string name as the key
)

ui_options = ParamDict()

sim_config_example = SimulationConfig(runSimulation_for_IPlotPDESols,
                                      shared_params,
                                      methods_dict,
                                      "Upwind",
                                      ui_options = ui_options
)

show1DSolutionFig(sim_config_example) 
# --- How it works in MakiePlotting ---
# 1. methods_obs uses keys from methods_dict ("EulerUpwind", "RK3MUSCL1", ...) for UI Checkboxes/Toggles.
# 2. createControls_Separated uses method_params_collection_obs (built from methods_dict)
#    to create UI elements for parameters listed in the *value* dictionaries (e.g., controls for
#    initFunc, regular, randomness_factor appear under "EulerUpwind" section).
# 3. Lift 1 gets the active method names (e.g., "EulerUpwind") from methods_obs[].
# 4. For each active method name, it calls assemble_params_for_run(... , "EulerUpwind").
# 5. assemble_params_for_run merges current shared_params_obs values with current
#    method_params_collection_obs["EulerUpwind"] values and adds "method" => "EulerUpwind".
# 6. This final dictionary is passed to runSimulation_for_IPlotPDESols.
