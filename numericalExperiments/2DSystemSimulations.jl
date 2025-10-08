include("../SimulationFunctions/runSystemSimulation.jl")

function _relax_velocities(a::Float64, N::Int)
    res = Vector{Vector{Tuple{Float64,Float64}}}(undef, N)
    for i = 1:N
        res[i] = [(a,0.),(-a,0.),(0.,a),(0.,-a)]
    end
    return res
end
function duplicateTuple(a::Any, N::Int)
    return Tuple([a for _ = 1:N])
end


clain_riemann_init = (
                      (0.029, 0.138,  1.206, 1.206),
                      (0.3,   0.5323, 0.,    1.206),
                      (1.5,   1.5,    0.,    0.   ),
                      (0.3,   0.5323, 1.206, 0.   )  
)

implosionInit = ((0.8, 1.0, 0.0, 0.0),
                 (1.0, 1.0, 0.0, 0.7276),
                 (1.0, 1.0, 0.7276, 0.0),
                 (0.5313, 0.4, 0.0, 0.0))

function main()
    function convertQuadrantInit(
        quadrant_data::NTuple{4, NTuple{4, T}}
    ) where T <: AbstractFloat
        
        # Helper function to convert primitive to conservative state
        function primToCons(rho, p, u, v)
            # Momentum components
            mx = rho * u
            my = rho * v
            
            # Total Energy (E = Internal_Energy + Kinetic_Energy)
            # Internal_Energy = p / (gamma - 1)
            # Kinetic_Energy = 0.5 * rho * (u^2 + v^2)
            E = p / (GAS_GAMMA_EULER - 1.0) + 0.5 * rho * (u^2 + v^2)
            
            return (rho, mx, my, E)
        end

        # Destructure the input tuple. 
        # Order: (BL, BR, TL, TR) for consistency with your example.
        (rho1, p1, u1, v1) = quadrant_data[1] # Bottom-Left (State 1)
        (rho2, p2, u2, v2) = quadrant_data[2] # Bottom-Right (State 2)
        (rho3, p3, u3, v3) = quadrant_data[3] # Top-Left (State 3)
        (rho4, p4, u4, v4) = quadrant_data[4] # Top-Right (State 4)

        # Convert each state to conservative variables
        state_BL = primToCons(rho1, p1, u1, v1)
        state_BR = primToCons(rho2, p2, u2, v2)
        state_TL = primToCons(rho3, p3, u3, v3)
        state_TR = primToCons(rho4, p4, u4, v4)

        # Combine into the required output format
        initial_states_vector = (state_BL, state_BR, state_TL, state_TR)
        
        return initial_states_vector
    end

# Example SimulationConfig for 2D Linear Advection
sim_config_2d = SimulationConfig(
    ParamDict(
        "tmax" => .3, "Nx" => 100, "Ny" => 100,
        "xmin" => 0., "xmax" => 1., "ymin" => 0., "ymax" => 1.,
        #"xmin" => -0.5, "xmax" => .5, "ymin" => -0.5, "ymax" => .5,
        "CFL" => 0.1, "snapshots" => 20, "interp_alpha" => 1.0,
        "interp_range" => 1.5,
        "init_func" => "q_riemann", # Use the new 2D function name
        #"init_params" => (0.,1.,(0.,0.),(1.,1.)),
        #"init_params" => (1.0, (0.0, 0.0), 1.5), # amp, center (x,y), width
        #"init_params" => (duplicateTuple(0.,4),duplicateTuple(1.,4),(0.,0.),(1.,0.)),
        "init_params" => (convertQuadrantInit(clain_riemann_init),(0.5,0.5)),
        #"init_params" => (convertQuadrantInit(implosionInit),(0.,0.)),
        "bc" => :outflow,
        "randomness_factor" => (0., 0.), # (x_rand, y_rand)
        "SEED" => 42,
        "sim_function" => "runSystemSimulation", # Point to the 2D run function
        "PDE" => "euler2d",
        "relax_velocities" => _relax_velocities(40.,4), "relax_epsilon" => 1e-6,
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
    "ARS222MUSCL2MOOD(Tiwari)"
    #["ARS222MUSCL2"]#,"ARS222MUSCL2MOOD(Tiwari)", "ARS222MUSCL2MOOD(Praveen)","ARS222Upwind"]
);

# --- How to run this with your IPlotPDESols package ---
# You would now pass `sim_config_2d` to your plotting functions.
# For example:
show2DSolutionFig(sim_config_2d;)
#show2DCutFig(sim_config_2d; scene_options = ParamDict("t"=>2., "line_vector" =>(1.,0.)))
# showConvergencePlot(sim_config_2d, "Nx", [20, 30, 40, 50]; ...)

### Single ParamDict for testing

# params = ParamDict(
#         "tmax" => 1., "Nx" => 10, "Ny" => 10,
#         "xmin" => -0.5, "xmax" => 0.5, "ymin" => -0.5, "ymax" => 0.5,
#         "CFL" => 0.4, "snapshots" => 20, "interp_alpha" => 1.0,
#         "save_relax" => false,
#         "interp_range" => 3.5,
#         "weight_function" => "exponential",
#         "init_func" => "q_riemann", # Use the new 2D function name
#         #"init_params" => (1.0, (0.0, 0.0), 1.5), # amp, center (x,y), width
#         #"init_params" => (duplicateTuple(0.,4),duplicateTuple(1.,4),(0.,0.),(1.,0.)),
#         "init_params" => implosionInit(),
#         "bc" => :fixed_dirichlet,
#         "randomness_factor" => (0.2, 0.2), # (x_rand, y_rand)
#         "SEED" => 42,
#         "sim_function" => "runSystemSimulation", # Point to the 2D run function
#         "PDE" => "euler2d",
#         "relax_velocities" => _relax_velocities(4.,4), "relax_epsilon" => 1e-6,
#         "timestepper" => "SimpleSplitting",
#         "main_gradient" => "MUSCL",
#         "order" => 2,
#         "main_flux" => "Rusanov",
#         "upwind_alg_2d" => "NonLinearPraveen",
#         "MOOD" => "none",
#         #"PDE_params" => (1.0, 1.0) # 2D velocity vector (vx, vy)
#     )
# runSystemSimulation(params);
# @profview runSystemSimulation(params);

end

main()
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