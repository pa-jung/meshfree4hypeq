include("../SimulationFunctions/runSystemSimulation.jl")

function _relax_velocities(a::Float64, N::Int)
    res = []
    for _ = 1:N
        push!(res, [(a,0),(-a,0),(0,a),(0,-a)])
    end
    return res
end
function duplicateTuple(a::Any, N::Int)
    return Tuple([a for _ = 1:N])
end

function implosionInit()
    # State 1 (Bottom-Left: x < 0, y < 0)
    rho1, p1, u1, v1 = 0.8, 1.0, 0.0, 0.0
    mx1 = rho1 * u1; my1 = rho1 * v1; E1 = p1/(GAS_GAMMA_EULER-1.0) + 0.5*rho1*(u1^2+v1^2)
    state_BL = (rho1, mx1, my1, E1)

    # State 2 (Bottom-Right: x >= 0, y < 0)
    rho2, p2, u2, v2 = 1.0, 1.0, 0.0, 0.7276
    mx2 = rho2 * u2; my2 = rho2 * v2; E2 = p2/(GAS_GAMMA_EULER-1.0) + 0.5*rho2*(u2^2+v2^2)
    state_BR = (rho2, mx2, my2, E2)

    # State 3 (Top-Left: x < 0, y >= 0)
    rho3, p3, u3, v3 = 1.0, 1.0, 0.7276, 0.0
    mx3 = rho3 * u3; my3 = rho3 * v3; E3 = p3/(GAS_GAMMA_EULER-1.0) + 0.5*rho3*(u3^2+v3^2)
    state_TL = (rho3, mx3, my3, E3)

    # State 4 (Top-Right: x >= 0, y >= 0)
    rho4, p4, u4, v4 = 0.5313, 0.4, 0.0, 0.0
    mx4 = rho4 * u4; my4 = rho4 * v4; E4 = p4/(GAS_GAMMA_EULER-1.0) + 0.5*rho4*(u4^2+v4^2)
    state_TR = (rho4, mx4, my4, E4)

    initial_states_vector = [state_BL, state_BR, state_TL, state_TR]
    center_point = (0.0, 0.0)
    return (initial_states_vector, center_point)
end
# Example SimulationConfig for 2D Linear Advection
sim_config_2d = SimulationConfig(
    ParamDict(
        "tmax" => 1., "Nx" => 15, "Ny" => 15,
        "xmin" => -0.5, "xmax" => 0.5, "ymin" => -0.5, "ymax" => 0.5,
        "CFL" => 0.4, "snapshots" => 20, "interp_alpha" => 1.0,
        "interp_range" => 3.5,
        "init_func" => "q_riemann", # Use the new 2D function name
        #"init_params" => (1.0, (0.0, 0.0), 1.5), # amp, center (x,y), width
        #"init_params" => (duplicateTuple(0.,4),duplicateTuple(1.,4),(0.,0.),(1.,0.)),
        "init_params" => implosionInit(),
        "bc" => :fixed_dirichlet,
        "randomness_factor" => (0.2, 0.2), # (x_rand, y_rand)
        "SEED" => 42,
        "sim_function" => "runSystemSimulation", # Point to the 2D run function
        "PDE" => "euler2d",
        "relax_velocities" => _relax_velocities(4.,4), "relax_epsilon" => 1e-6,
        "save_relax" => false, "weight_function" => "exponential"
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
        "ARS222MUSCL2MOOD(Praveen)" => ParamDict(
            "timestepper" => "ARS222",
            "main_gradient" => "MUSCL",
            "order" => 2,
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "fallback_gradient" => "Upwind", "upwind_alg_2d" => "Praveen",
            "MOOD" => "U2", "delta_relax" => 0.,
        ), 
        "ARS222MUSCL2MOOD(Tiwari)" => ParamDict(
            "timestepper" => "ARS222",
            "main_gradient" => "MUSCL",
            "order" => 2,
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "fallback_gradient" => "Upwind", "upwind_alg_2d" => "Tiwari",
            "MOOD" => "U2","delta_relax" => 0.,
        ),            
    ),
    #"ARS222Upwind"
    ["ARS222MUSCL2","ARS222MUSCL2MOOD(Tiwari)", "ARS222MUSCL2MOOD(Praveen)","ARS222Upwind"]
);

# --- How to run this with your IPlotPDESols package ---
# You would now pass `sim_config_2d` to your plotting functions.
# For example:
#show2DSolutionFig(sim_config_2d;)
#show2DCutFig(sim_config_2d; scene_options = ParamDict("t"=>2., "line_vector" =>(1.,1.)))
# showConvergencePlot(sim_config_2d, "Nx", [20, 30, 40, 50]; ...)

### Single ParamDict for testing

params = ParamDict(
        "tmax" => 1., "Nx" => 10, "Ny" => 10,
        "xmin" => -0.5, "xmax" => 0.5, "ymin" => -0.5, "ymax" => 0.5,
        "CFL" => 0.4, "snapshots" => 20, "interp_alpha" => 1.0,
        "save_relax" => false,
        "interp_range" => 3.5,
        "weight_function" => "exponential",
        "init_func" => "q_riemann", # Use the new 2D function name
        #"init_params" => (1.0, (0.0, 0.0), 1.5), # amp, center (x,y), width
        #"init_params" => (duplicateTuple(0.,4),duplicateTuple(1.,4),(0.,0.),(1.,0.)),
        "init_params" => implosionInit(),
        "bc" => :fixed_dirichlet,
        "randomness_factor" => (0.2, 0.2), # (x_rand, y_rand)
        "SEED" => 42,
        "sim_function" => "runSystemSimulation", # Point to the 2D run function
        "PDE" => "euler2d",
        "relax_velocities" => _relax_velocities(4.,4), "relax_epsilon" => 1e-6,
        "timestepper" => "ARS222",
        "main_gradient" => "MUSCL",
        "order" => 2,
        "main_flux" => "Rusanov",
        "MOOD" => "none",
        #"PDE_params" => (1.0, 1.0) # 2D velocity vector (vx, vy)
    )

@profview runSystemSimulation(params);
#runSystemSimulation(params);
#using Test # You might need to run `using Pkg; Pkg.add("Test")` if not in a test environment.

# Assume your modules are loaded, e.g.:
# using .HyperbolicPDEs
# using .SourceTerms

# --- 1. Create dummy objects to build a valid `RelaxationSourceTerm` ---

# Assume your modules are loaded
# using .HyperbolicPDEs
# using .SourceTerms

# Assume your modules are loaded
# using .HyperbolicPDEs
# using .SourceTerms

# Assume your modules are loaded
# using .HyperbolicPDEs
# using .SourceTerms

# println("--- Starting Debug Script using the `code_warntype` function ---")

# try
#     # --- 1. Setup (same as before) ---
#     euler_eq = Euler2D() 
#     sample_maxwellian = MaxwellianFunctor(euler_eq, 1, 1, 1.0, 0.25, 2.0)
#     maxwellians_vec = [sample_maxwellian, sample_maxwellian]
#     epsilon = 0.01
#     kinetic_indices = [[1], [2]]
#     rs = RelaxationSourceTerm(maxwellians_vec, epsilon, kinetic_indices)
#     num_components = 2
#     S_out = zeros(Float64, num_components)
#     U_kinetic = ones(Float64, num_components)
#     pos = 0.0
#     time = 0.0

#     println("Step 1: Setup complete.")

#     # --- 2. Use the `code_warntype` function directly ---
#     println("Step 2: Calling `code_warntype` function to capture output...")

#     buffer = IOBuffer()

#     # Get the types of the arguments for the function call
#     arg_types = (typeof(S_out), typeof(U_kinetic), typeof(pos), typeof(time))

#     # Call the function directly, passing the buffer as the output destination,
#     # the function to analyze (our functor `rs`), and the types of its arguments.
#     code_warntype(buffer, rs, arg_types)

#     output_string = String(take!(buffer))

#     println("\n--- COMPILER ANALYSIS ---")
#     println(output_string)
#     println("--- END OF ANALYSIS ---")

# catch e
#     println("\nERROR: An error occurred.")
#     showerror(stdout, e, catch_backtrace())
#     println()
# end