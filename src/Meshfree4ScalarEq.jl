module Meshfree4ScalarEq

export runScalarSimulation, runSystemSimulation, GAS_GAMMA_EULER

# Imports
using Random
using Logging



# --- NEW: Add this function at the end of your module ---
function __init__()
    # This code will run once when the module is loaded.
    # It sets the logger for the entire application.
    min_level_to_show = Logging.Warn
    global_logger(ConsoleLogger(stderr, min_level_to_show))
    println("Logger initialized to show ",min_level_to_show,"-Level.")
end
# ---------------------------------------------------------

# Project wide random generator object
global const SEED = 10
global const rng = MersenneTwister(SEED)

# Include submodules
include("HyperbolicPDEs.jl")
using .HyperbolicPDEs

include("FluxFunctions.jl")
using .FluxFunctions

include("MLSWeightFunctions.jl")
using .MLSWeightFunctions

include("SimSettings.jl")
using .SimSettings

include("ParticleGrids.jl")
using .ParticleGrids

include("InitialConditions.jl")
using .InitialConditions

include("MOOD.jl")
using .MOOD

include("GridManagement.jl")
using .GridManagement

include("Interpolations.jl")
using .Interpolations

include("SourceTerms.jl")
using .SourceTerms

include("ImplicitSolvers.jl")
using .ImplicitSolvers

include("TimeIntegration.jl")
using .TimeIntegration

include("ParticleGridStability.jl")
using .ParticleGridStability

using IPlotPDESols

include("../SimulationFunctions/runScalarSimulation.jl")
include("../SimulationFunctions/runSystemSimulation.jl")

end  # module 