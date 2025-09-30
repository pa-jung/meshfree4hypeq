using IPlotPDESols # or just `const ParamDict = Dict{String, Any}` if not using the package

# This combined ParamDict contains all the necessary keys to run a single
# 2D simulation using your `runScalar2DSim` function.

params = ParamDict(
    # --- Shared Parameters ---
    "tmax" => 3.0,
    "Nx" => 50,
    "Ny" => 50,
    "xmin" => -5.0,
    "xmax" => 5.0,
    "ymin" => -5.0,
    "ymax" => 5.0,
    "CFL" => 0.1,
    "snapshots" => 20,
    "interp_alpha" => 1.0,
    "interp_range" => 3.5,
    "init_func" => "box", # Using the unified struct
    #"init_params" => (1.0, (0.0, 0.0), 1.5), # (amplitude, (centerX, centerY), width)
    "init_params" => (0.,1.,-2.,2.,-2.,2.),
    "randomness_factor" => (0., 0.), # (x_rand_factor, y_rand_factor)
    "SEED_value" => 42,
    "PDE" => "linear",
    "PDE_params" => (1.0, 0.), # 2D velocity vector (vx, vy)
    "bc" => :outflow,
    "sim_function" => "runScalarSimulation",
    "weight_function" => "exponential",
    "limiter" => "VK",

    # --- Method-Specific Parameters for "RK4-MUSCL2-2D" ---
    "timestepper" => "RalstonRK2",
    "main_gradient" => "MUSCL",
    "order" => 2,
    "upwind_alg_2d" => "Classic",
    "main_flux" => "Rusanov",
    #"mood" => "U2", "delta_relax" => 0. 
    #"mood" => "none", # No MOOD for this run
)

# params = ParamDict(
#     # --- Shared Parameters ---
#     "tmax" => 10.,
#     "N" => 100,
#     "xmin" => -5.0,
#     "xmax" => 5.0,
#     "CFL" => .2,
#     #"dt" => .25,
#     "snapshots" => 20,
#     "interp_alpha" => 1.0,
#     "interp_range" => 3.5,
#     "init_func" => "box", # Using the unified struct
#     #"init_params" => (1.0, 0., 1.5), # (amplitude, (centerX, centerY), width)
#     "init_params" => (0.,1.,-2.,2.),
#     "randomness_factor" => 0.2, # (x_rand_factor, y_rand_factor)
#     "SEED_value" => 42,
#     "PDE" => "linear",
#     "weight_function" => "exponential",
#     "PDE_params" => 1.0, # 2D velocity vector (vx, vy)
#     "bc" => :outflow,
#     "sim_function" => "runScalarSimulation",

#     # --- Method-Specific Parameters for "RK4-MUSCL2-2D" ---
#     "timestepper" => "RalstonRK2",
#     "main_gradient" => "WENO",
#     "fallback_gradient" => "Upwind",
#     "fallback_flux" => "Rusanov",
#     "order" =>2,
#     "main_flux" => "Rusanov",
#     #"limiter" => "superbee",
#     #"MOOD" => "U2", "delta_relax" => 0. # No MOOD for this run
# )

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
        "TestMUSCL(superbee)" => ParamDict("limiter" => "superbee"),
        "TestMUSCL(minmod)" => ParamDict("limiter" => "minmod"),
        "TestMUSCL(VK)" => ParamDict("limiter" => "VK"),
        "Test" => ParamDict(),
        "TestMUSCL(SmSw)" => ParamDict("timestepper" => "RalstonRK2SmoothSwitch", "switch_tol" => 5.),
        "TestMUSCLRelax" => ParamDict("timestepper" => "PRSSP3", "relax_velocities" => [-1.,1.], "save_relax" => false, "relax_epsilon" => 1e-6),
    ),
    #"all"
    "TestMUSCL(VK)"

);
#show1DSolutionFig(test_config)
scene_options = Dict{String, Any}("line_vector" => (1.,0.), "deviation" => 2)
show2DCutFig(test_config;scene_options = scene_options)

