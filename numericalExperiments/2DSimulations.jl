include("../SimulationFunctions/runScalar2DSim.jl")

# Example SimulationConfig for 2D Linear Advection
sim_config_2d = SimulationConfig(
    ParamDict(
        "tmax" => 2.0, "Nx" => 40, "Ny" => 40,
        "xmin" => -5.0, "xmax" => 5.0, "ymin" => -5.0, "ymax" => 5.0,
        "CFL" => 0.4, "snapshots" => 20, "interp_alpha" => 1.0,
        "interp_range" => 3.5,
        "init_func" => "riemann", # Use the new 2D function name
        #"init_params" => (1.0, (0.0, 0.0), 1.5), # amp, center (x,y), width
        #"init_params" => (0.,1.,-2.,2.,-2.,2.),
        "init_params" => (1.,0.,(0.,0.),(1.,1.)),
        "bc" => :outflow,
        "randomness_factor" => (0.2, 0.2), # (x_rand, y_rand)
        "SEED_value" => 42,
        "sim_function" => "runScalar2DSim", # Point to the 2D run function
        "PDE" => "linear",
        "PDE_params" => (2.0, -0.5) # 2D velocity vector (vx, vy)
    ),
    MethodDict(
        "RK4-MUSCL2-2D" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "order" => 2,
            "main_flux" => "Rusanov",
            "MOOD" => "none"
        ),
        "RK4-MUSCL2-2D-MOOD" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "order" => 2,
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "fallback_gradient" => "Upwind", "upwind_alg_2d" => "Praveen",
            "MOOD" => "U2", "delta_relax" => 0.
        ),
        "Analytical Solution" => ParamDict(
            "ignore" => ["interp_range", "interp_alpha", "randomness_factor", "SEED", "order", "relax_velocities"]
        ),
        "RK2-Upwind-Praveen-2D" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "Upwind",
            "order" => 1,
            "main_flux" => "Rusanov",
            "upwind_alg_2d" => "Praveen", # Select a specific 2D upwind algorithm
            "MOOD" => "none"
        ),
        "RK4-Central-2D" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "Central",
            "order" => 2,
            "MOOD" => "none"
        )
    ),
    ["RK4-MUSCL2-2D-MOOD","Analytical Solution"] # Methods to run by default
);

# --- How to run this with your IPlotPDESols package ---
# You would now pass `sim_config_2d` to your plotting functions.
# For example:
#show2DSolutionFig(sim_config_2d;)
show2DCutFig(sim_config_2d; scene_options = ParamDict("t"=>2.))
# showConvergencePlot(sim_config_2d, "Nx", [20, 30, 40, 50]; ...)