using IPlotPDESols # or just `const ParamDict = Dict{String, Any}` if not using the package

# This combined ParamDict contains all the necessary keys to run a single
# 2D simulation using your `runScalar2DSim` function.

params = ParamDict(
    # --- Shared Parameters ---
    "tmax" => 1.0,
    "Nx" => 10,
    "Ny" => 10,
    "xmin" => -5.0,
    "xmax" => 5.0,
    "ymin" => -5.0,
    "ymax" => 5.0,
    "CFL" => 0.4,
    "snapshots" => 20,
    "interp_alpha" => 1.0,
    "interp_range" => 2.5,
    "init_func" => "gauss", # Using the unified struct
    "init_params" => (1.0, (0.0, 0.0), 1.5), # (amplitude, (centerX, centerY), width)
    "randomness_factor" => (0.0, 0.0), # (x_rand_factor, y_rand_factor)
    "SEED_value" => 42,
    "PDE" => "linear",
    "PDE_params" => (1.0, 0.5), # 2D velocity vector (vx, vy)

    # --- Method-Specific Parameters for "RK4-MUSCL2-2D" ---
    "timestepper" => "RalstonRK2",
    "main_gradient" => "MUSCL",
    "order" => 2,
    "main_flux" => "Rusanov",
    "mood" => "none", # No MOOD for this run
    "fallback_gradient" => nothing, # No fallback needed if MOOD is off
    "fallback_flux" => nothing
)

# --- How to use this for testing ---
#
# 1. Make sure all your project modules and the `runScalar2DSim.jl` file are loaded.
#
# 2. You can then call your function directly:
#
include("../SimulationFunctions/runScalar2DSim.jl")
sim_data = runScalar2DSim(params)
#
# 3. `sim_data` will now hold the results (a SimData2D object), which you can inspect.
#
