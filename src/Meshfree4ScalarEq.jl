module Meshfree4ScalarEq

# Imports
using Random
using Logging

# --- NEW: Add this function at the end of your module ---
function __init__()
    # This code will run once when the module is loaded.
    # It sets the logger for the entire application.
    min_level_to_show = Logging.Info
    global_logger(ConsoleLogger(stderr, min_level_to_show))
    println("IPlotPDESols logger initialized to show warnings and errors only.")
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

include("SimSettings.jl")
using .SimSettings

include("ParticleGrids.jl")
using .ParticleGrids

include("Interpolations.jl")
using .Interpolations

include("SourceTerms.jl")
using .SourceTerms

include("ImplicitSolvers.jl")
using .ImplicitSolvers

include("MOOD.jl")
using .MOOD

include("TimeIntegration.jl")
using .TimeIntegration

include("ParticleGridStability.jl")
using .ParticleGridStability

include("InitialConditions.jl")
using .InitialConditions

end  # module 