using Meshfree4ScalarEq
using IPlotPDESols

function _relax_velocities(a::Float64, N::Int)
    res = []
    for _ = 1:N
        push!(res, [(a,0),(-a,0),(0,a),(0,-a)])
    end
    return res
end

function main()
    # Example SimulationConfig for 2D Linear Advection
    sim_config_2d = SimulationConfig(
        runScalarSimulation,
        ParamDict(
            "tmax" => 3.0, "Nx" => 200, "Ny" => 200,
            "xmin" => -5.0, "xmax" => 5.0, "ymin" => -5.0, "ymax" => 5.0,
            "CFL" => 0.4, "snapshots" => 50, "interp_alpha" => 1.0,
            "interp_range" => 3.5, "remove_ghosts" => true,
            "init_func" => "riemann", # Use the new 2D function name
            #"init_params" => (1.0, (0.0, 0.0), 1.5), # amp, center (x,y), width
            #"init_params" => (0.,1.,-2.,2.,-2.,2.),
            "init_params" => (-.5,.6,(0.,0.),(1.,1.)),
            "bc" => :outflow,
            "randomness_factor" => (0.2, 0.2), # (x_rand, y_rand)
            "SEED" => 42,
            "weight_function" => "exponential",
            "sim_function" => "runScalarSimulation", # Point to the 2D run function
            "PDE" => "burgers2d",
            #"PDE_params" => (1.0, 1.0) # 2D velocity vector (vx, vy)
        ),
        MethodDict(
            "RK2MUSCL2" => ParamDict(
                "timestepper" => "RalstonRK2",
                "main_gradient" => "MUSCL",
                "order" => 2,
                "main_flux" => "Rusanov",
                "MOOD" => "none"
            ),
            "RK2MUSCL2MOOD" => ParamDict(
                "timestepper" => "RalstonRK2",
                "main_gradient" => "MUSCL",
                "order" => 2,
                "main_flux" => "Rusanov",
                "fallback_flux" => "Rusanov",
                "fallback_gradient" => "Upwind", "upwind_alg_2d" => "Praveen",
                "MOOD" => "U2", "delta_relax" => 0.
            ),
            "RK2MUSCL2Limiter" => ParamDict(
                "timestepper" => "RalstonRK2",
                "main_gradient" => "MUSCL",
                "order" => 2,
                "main_flux" => "Rusanov",
                "MOOD" => "none",
                "limiter" => "VK",
            ),
            "RK2Upwind" => ParamDict(
                "timestepper" => "RalstonRK2",
                "main_gradient" => "Upwind",
                "upwind_alg_2d" => "Classic",
                "order" => 1,
                "main_flux" => "Rusanov",
                "MOOD" => "none"
            ),
            "RK2Upwind(Tiwari)" => ParamDict(
                "timestepper" => "RalstonRK2",
                "main_gradient" => "Upwind",
                "upwind_alg_2d" => "Tiwari",
                "order" => 1,
                "main_flux" => "Rusanov",
                "MOOD" => "none"
            ),

            "Analytical Solution" => ParamDict(
                "ignore" => ["weight_function","interp_range", "interp_alpha", "randomness_factor", "SEED", "order", "relax_velocities", "remove_ghosts"]
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
            ),
            "ARS222MUSCL2" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "MUSCL",
                "order" => 2,
                "main_flux" => "Rusanov",
                "MOOD" => "none",
                "relax_velocities" => (_relax_velocities(2.,1)[1],), "relax_epsilon" => 1e-6,
                "save_relax" => false,
            ),    
            "ARS222MUSCL2Limiter" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "MUSCL",
                "order" => 2,
                "main_flux" => "Rusanov",
                "MOOD" => "none", "limiter" => "VK",
                "relax_velocities" => (_relax_velocities(2.,1)[1],), "relax_epsilon" => 1e-6,
                "save_relax" => false,
            ),    
            "ARS222Upwind(Tiwari)" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "Upwind",
                "order" => 1,
                "main_flux" => "Rusanov",
                "upwind_alg_2d" => "Tiwari",
                "MOOD" => "none",
                "relax_velocities" => _relax_velocities(2.,1), "relax_epsilon" => 1e-6,
                "save_relax" => false,
            ),
            "ARS222Upwind(Classic)" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "Upwind",
                "order" => 1,
                "main_flux" => "Rusanov",
                "upwind_alg_2d" => "Classic",
                "MOOD" => "none",
                "relax_velocities" => _relax_velocities(2.,1), "relax_epsilon" => 1e-6,
                "save_relax" => false,
            ),
            "ARS222Upwind" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "Upwind",
                "order" => 1,
                "main_flux" => "Rusanov",
                "upwind_alg_2d" => "Tiwari",
                "MOOD" => "none",
                "relax_velocities" => _relax_velocities(2.,1), "relax_epsilon" => 1e-6,
                "save_relax" => false,
            ),    
            "ARS222MUSCL2MOOD" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "MUSCL",
                "order" => 2,
                "main_flux" => "Rusanov",
                "fallback_flux" => "Rusanov",
                "fallback_gradient" => "Upwind", "upwind_alg_2d" => "Tiwari",
                "MOOD" => "U2", "delta_relax" => 0.,
                "relax_velocities" => _relax_velocities(2.,1), "relax_epsilon" => 1e-6,
                "save_relax" => false,
            ), 
            "ARS222MUSCL2TotalFallback" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "MUSCL",
                "order" => 2,
                "main_flux" => "Rusanov",
                "fallback_flux" => "Rusanov",
                "fallback_gradient" => "Upwind", "upwind_alg_2d" => "Praveen",
                "MOOD" => "only",
                "relax_velocities" => _relax_velocities(2.,1), "relax_epsilon" => 1e-6,
                "save_relax" => false,
            ),            
        ),
        #["ARS222MUSCL2Limiter","Analytical Solution"]
        #"RK2MUSCL2"
        #["Analytical Solution","RK2Upwind(Classic)", "RK2Upwind(Tiwari)", "ARS222Upwind(Classic)", "ARS222Upwind(Tiwari)"]
        ["Analytical Solution", "RK2MUSCL2","RK2MUSCL2MOOD","RK2MUSCL2Limiter","RK2Upwind"]
        #["Analytical Solution","ARS222MUSCL2","ARS222MUSCL2MOOD","ARS222Upwind","ARS222MUSCL2Limiter"] # Methods to run by default
    );
    params = ParamDict(
            "tmax" => 2.0, "Nx" => 300, "Ny" => 300,
            "xmin" => -6.0, "xmax" => 6.0, "ymin" => -6.0, "ymax" => 6.0,
            "CFL" => 0.4, "snapshots" => 20, "interp_alpha" => 1.0,
            "interp_range" => 3.5,
            "init_func" => "riemann", # Use the new 2D function name
            #"init_params" => (1.0, (0.0, 0.0), 1.5), # amp, center (x,y), width
            #"init_params" => (0.,1.,-2.,2.,-2.,2.),
            "init_params" => (0.,1.,(0.,0.),(1.,1.)),
            "bc" => :outflow,
            "randomness_factor" => (0.2, 0.2), # (x_rand, y_rand)
            "SEED" => 42,
            "weight_function" => "exponential",
            "sim_function" => "runScalarSimulation", # Point to the 2D run function
            "PDE" => "burgers2d",
            #"PDE_params" => (1.0, 1.0) # 2D velocity vector (vx, vy)
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "order" => 2,
            "main_flux" => "Rusanov",
            "MOOD" => "none"
        )

    return params, sim_config_2d
end

params, sim_config_2d = main()

#@profview runScalarSimulation(params)
# For example:
#show2DSolutionFig(sim_config_2d;)
showDynamicDependence(sim_config_2d;ui_options  = :publication)
#show2DCutFig(sim_config_2d; scene_options = ParamDict("t"=>5., "line_vector" =>(1.,1.), "deviation" => 1.), ui_options  = :publication)
# showConvergencePlot(sim_config_2d, "Nx", [20, 30, 40, 50]; ...)