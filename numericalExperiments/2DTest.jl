using IPlotPDESols # or just `const ParamDict = Dict{String, Any}` if not using the package

# This combined ParamDict contains all the necessary keys to run a single
# 2D simulation using your `runScalar2DSim` function.

# params = ParamDict(
#     # --- Shared Parameters ---
#     "tmax" => 2.0,
#     "Nx" => 40,
#     "Ny" => 40,
#     "xmin" => -5.0,
#     "xmax" => 5.0,
#     "ymin" => -5.0,
#     "ymax" => 5.0,
#     "CFL" => 0.4,
#     "snapshots" => 20,
#     "interp_alpha" => 1.0,
#     "interp_range" => 2.5,
#     "init_func" => "gauss", # Using the unified struct
#     "init_params" => (1.0, (0.0, 0.0), 1.5), # (amplitude, (centerX, centerY), width)
#     "randomness_factor" => (0.0, 0.0), # (x_rand_factor, y_rand_factor)
#     "SEED_value" => 42,
#     "PDE" => "linear",
#     "PDE_params" => (1.0, 0.5), # 2D velocity vector (vx, vy)
#     "bc" => :periodic,

#     # --- Method-Specific Parameters for "RK4-MUSCL2-2D" ---
#     "timestepper" => "RalstonRK2",
#     "main_gradient" => "Upwind",
#     "order" => 1,
#     "upwind_alg_2d" => 
#     "main_flux" => "Rusanov",
#     "mood" => "none", # No MOOD for this run
# )

params = ParamDict(
    # --- Shared Parameters ---
    "tmax" => 2.0,
    "N" => 100,
    "xmin" => -5.0,
    "xmax" => 5.0,
    "CFL" => 0.4,
    "snapshots" => 20,
    "interp_alpha" => 1.0,
    "interp_range" => 2.5,
    "init_func" => "gauss", # Using the unified struct
    "init_params" => (1.0, 0., 1.5), # (amplitude, (centerX, centerY), width)
    "randomness_factor" => 0., # (x_rand_factor, y_rand_factor)
    "SEED_value" => 42,
    "PDE" => "linear",
    "weight_function" => "exponential",
    "PDE_params" => 1.0, # 2D velocity vector (vx, vy)
    "bc" => :periodic,
    "sim_function" => "runScalarSimulation",

    # --- Method-Specific Parameters for "RK4-MUSCL2-2D" ---
    "timestepper" => "RalstonRK2",
    "main_gradient" => "Upwind",
    "order" => 1,
    "main_flux" => "Rusanov",
    "mood" => "none", # No MOOD for this run
)
# --- How to use this for testing ---
#
# 1. Make sure all your project modules and the `runScalar2DSim.jl` file are loaded.
#
# 2. You can then call your function directly:
#
include("../SimulationFunctions/runScalarSimulation.jl")
#sim_data = runScalarSimulation(params)

# 3. `sim_data` will now hold the results (a SimData2D object), which you can inspect.
#
test_config = SimulationConfig(
    params,
    MethodDict(
        "TestUpwind" => ParamDict()
    ),
    "TestUpwind"

)
show1DSolutionFig(test_config)