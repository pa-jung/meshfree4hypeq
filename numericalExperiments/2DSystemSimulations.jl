include("../SimulationFunctions/runSystem2DSim.jl")

function _relax_velocities(a::Float64, N::Int)
    res = []
    for _ = 1:N
        push!(res, [(a,0),(-a,0),(0,a),(0,-a)])
    end
    return res
end

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
        "init_params" => (0.,1.,(0.,0.),(1.,1.)),
        "bc" => :outflow,
        "randomness_factor" => (0.2, 0.2), # (x_rand, y_rand)
        "SEED_value" => 42,
        "sim_function" => "runSystem2DSim", # Point to the 2D run function
        "PDE" => "euler2d",
        "relax_velocities" => _relax_velocities(4.,4), "relax_epsilon" => 1e-6 
        #"PDE_params" => (1.0, 1.0) # 2D velocity vector (vx, vy)
    ),
    MethodDict(
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
        "ARS222MUSCL2" => ParamDict(
            "timestepper" => "ARS222",
            "main_gradient" => "MUSCL",
            "order" => 2,
            "main_flux" => "Rusanov",
            "MOOD" => "none",

        ),    
        "ARS222Upwind" => ParamDict(
            "timestepper" => "ARS222",
            "main_gradient" => "Upwind",
            "order" => 1,
            "main_flux" => "Rusanov",
            "upwind_alg_2d" => "Praveen",
            "MOOD" => "none",
        ),    
        "ARS222MUSCL2MOOD" => ParamDict(
            "timestepper" => "ARS222",
            "main_gradient" => "MUSCL",
            "order" => 2,
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "fallback_gradient" => "Upwind", "upwind_alg_2d" => "Praveen",
            "MOOD" => "U2", "delta_relax" => 0.,
        ), 
        "ARS222MUSCL2TotalFallback" => ParamDict(
            "timestepper" => "ARS222",
            "main_gradient" => "MUSCL",
            "order" => 2,
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "fallback_gradient" => "Upwind", "upwind_alg_2d" => "Praveen",
            "MOOD" => "only",
        ),            
    ),
    "ARS222Upwind"
    #["ARS222MUSCL2","ARS222MUSCL2MOOD","ARS222Upwind"]
    #["ARS222MUSCL2","ARS222MUSCL2MOOD","ARS222MUSCL2TotalFallback","RK2MUSCL2","RK2MUSCL2MOOD","ARS222Upwind"] # Methods to run by default
);

# --- How to run this with your IPlotPDESols package ---
# You would now pass `sim_config_2d` to your plotting functions.
# For example:
#show2DSolutionFig(sim_config_2d;)
show2DCutFig(sim_config_2d; scene_options = ParamDict("t"=>2., "line_vector" =>(1.,1.)))
# showConvergencePlot(sim_config_2d, "Nx", [20, 30, 40, 50]; ...)