include("../SimulationFunctions/runScalarSim.jl")

# Example SimulationConfig for Burgers
sim_config_burgers = SimulationConfig(
    runScalarSim, # Use the new runner

    ParamDict(
        "tmax" => 5, "N" => 100, "xmin" => -5.0, "xmax" => 5.0,
        "CFL" => .2, "snapshots" => 5, "interp_alpha" => 1.0,
        "interp_range" => 3.5,
        "init_func" => "box",
        "init_params" => (0., 1., -2., 2.),#(0., 1., -4., -2.), #
        "randomness_factor" => 0.2, # Provide default needed when regular=false
        "SEED" => 10, 
        "bc" => :periodic,
        "order" => 1, "PDE" => "linear", "PDE_params" => (1.,)
    ),

    MethodDict(
        "RK2Upwind" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
        ),
        "RK2MUSCL2Smooth" => ParamDict(
            "timestepper" => "RalstonRK2SmoothSwitch",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "switch_tol" => 1e-8,
            "delta_relax" => 0.,
            "order" => 2
        ),
        "RK2MUSCL2(VKLimiter)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "limiter" => "VK",
            "MOOD" => "none",
        ),
        "RK2MUSCL2(Superbee)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "limiter" => "superbee",
            "MOOD" => "none",
        ),
        "RK2MUSCL2MOOD(U2)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(U2Relax)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 1.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(LoubertU2)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "LoubertU2",
            "delta_relax" => 1.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(U1)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U1",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(U1Relax)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U1",
            "delta_relax" => 5.,
            "order" => 2,
        ),
        "RK2MUSCL2" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 2
        ),
        "RK4MUSCL2" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 2,
        ),
        "RK4MUSCL5" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "MOOD" => "none",
        ),
        "RK4MUSCL5MOOD" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 5
        ),
        "EulerUpwind" => ParamDict(
            "timestepper" => "EulerUpwind",
            "main_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "order" => 1
        ),
        "Analytic Solution" => ParamDict(
            "N" => (:const, 1000), 
            "ignore" => ["interp_alpha", "randomness_factor", "SEED", "order"]
             # No randomness_factor needed when regular=true
        ),
        "LLF(uniform grid)" => ParamDict(
            "randomness_factor" => (:const,0.),
            "main_flux" => "Rusanov",
            "order" => 1,
            "timestepper" => "Classic",
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
                "ARS233MUSCL5" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "relax_method" => true,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none",
        ),
                "ARS233MUSCL5MOOD" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "relax_method" => true,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.,
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        # "PRSSP3MUSCL5" => ParamDict(
        #     "timestepper" => "PRSSP3",
        #     "main_gradient" => "MUSCL",
        #     "main_flux" => "Rusanov",
        #     "order" => 5,
        #     "relax_method" => true,
        #     "relax_velocities" => (1.,-1.),
        #     "relax_epsilon" => 10. ^ -8,
        #     "MOOD" => "none",
        # ),
        "ARS233MUSCL2" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "relax_method" => true,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none",
        ),
        "LWMOOD" => ParamDict(
            "timestepper" => "LW",
            "order" => 2,
            "MOOD" => "U1", "delta_relax" => 0.,
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
        "LW" => ParamDict(
            "timestepper" => "LW",
            "order" => 2,
            "MOOD" => "none",
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
        "ARS233MUSCL2MOOD" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "relax_method" => true,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.,
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        # "PRSSP3MUSCL2" => ParamDict(
        #     "timestepper" => "PRSSP3",
        #     "main_gradient" => "MUSCL",
        #     "main_flux" => "Rusanov",
        #     "MOOD" => "none",
        #     "order" => 2,
        #     "relax_method" => true,
        #     "relax_velocities" => (1.,-1.),
        #     "relax_epsilon" => 10. ^ -8,
        # )

    ),
    ["RK2MUSCL2","RK2MUSCL2MOOD(U2)", "ARS233MUSCL2MOOD"]#,"LW", "EulerUpwind"]
    #["LWMOOD","RK2MUSCL2", "LW","ARS233MUSCL2MOOD", "Analytic Solution", "RK2MUSCL2MOOD(U1)", "RK2MUSCL2MOOD(U2)", "RK4MUSCL5MOOD"]
    #["LWMOOD","RK2MUSCL2", "LW","ARS233MUSCL2MOOD", "Analytic Solution", "RK2MUSCL2MOOD(U1)", "RK2MUSCL2MOOD(U2)", "RK2MUSCL2MOOD(U2Relax)", "RK2MUSCL2MOOD(U1Relax)", "RK4MUSCL5MOOD"]#["RK2MUSCL2MOOD", "RK2MUSCL2", "RK4MUSCL5MOOD", "Analytic Solution"] #, "Relax Method 2", "Relax Method 3rd order","Classic","SlopeLimiter","SmoothSwitching","Regular MOOD", "OnlyFallback"]
    #["RK2MUSCL2Smooth", "Analytic Solution"]
    #["RK2MUSCL2MOOD(U1)", "Analytic Solution"]
);

# Pass this config to your IPlotPDESols functions
show1DSolutionFig(sim_config_burgers; calc_stats = false, ui_options = :publication);
#showDynamicDependence(sim_config_burgers; ui_options = :publication)
#calculateConvergenceData(sim_config_burgers, "N", 10. .^(1:.25:2); force_int_param = true)
#showConvergencePlot(sim_config_burgers, "N", 10. .^(1.5:.25:2); force_int_param = true, initial_calc = true, ui_options = :publication)
#showConvergencePlot(sim_config_burgers, "delta_relax", 10. .^(0.:0.05:1.5); force_int_param = false, initial_calc = false, ui_options = :publication)
#showConvergencePlot(sim_config_burgers, "switch_tol", 10. .^(-5:.1:-2); force_int_param = false, initial_calc = false, ui_options = :publication)
#showConvergencePlot(sim_config_burgers, "SEED", range(1,10000,5); force_int_param = true, initial_calc = true, ui_options = :publication);

