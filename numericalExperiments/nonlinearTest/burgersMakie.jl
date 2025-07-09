# Place this code where your other simulation runner function is defined,
# ensuring access to necessary modules and types.

using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq.FluxFunctions
using Meshfree4ScalarEq.SourceTerms
using Meshfree4ScalarEq.ImplicitSolvers
using Meshfree4ScalarEq.PlottingUtils
using Random                    # For RNG state copy
using IPlotPDESols
using Meshfree4ScalarEq
# --- Keep initial condition function definitions ---
function sineInit(x::Real, a::Real, b::Real, c::Real) return a*sin(2*pi*x/b) + c end # Adjusted from burgers.txt
function gaussInit(x::Real, a::Real, b::Real, width::Real) return a*exp(-((x-b)/width)^2) end
function riemannInit(x::Real, uL::Real, uR::Real, x0::Real)::Float64
    return x < x0 ? uL : uR
end

"""
    boxInit(x::Real, u_background::Real, u_box::Real, x_box_start::Real, x_box_end::Real)::Float64

Generates a box-like initial condition.
Returns `u_box` if `x_box_start <= x <= x_box_end`, and `u_background` otherwise.
"""
function boxInit(x::Real, u_background::Real, u_box::Real, x_box_start::Real, x_box_end::Real)::Float64
    if x_box_start <= x <= x_box_end
        return u_box
    else
        return u_background
    end
end

# For Linear Advection (handles both periodic and fixed/outflow BCs)
function linearSolution(x::Real, t::Real, eq::LinearAdvection, initFunc::Function, init_params::Tuple, xmin::Real, xmax::Real; bc::Symbol = :periodic)
    # Find the characteristic foot x0 = x - a*t
    x0 = x - eq.vel * t
    
    if bc == :periodic
        # Map the characteristic foot back into the periodic domain
        domain_length = xmax - xmin
        x0_mapped = xmin + mod(x0 - xmin, domain_length)
        return initFunc(x0_mapped, init_params...)
    else # :fixed or :outflow
        # For an infinite domain assumption, just evaluate at the shifted position
        return initFunc(x0, init_params...)
    end
end



"""
    riemannInitAna(x::Real, t::Real, uL::Real, uR::Real, x0::Real, eq::BurgersEquation)

Provides the exact solution to the 1D Burger's Riemann problem.
It automatically handles both shock (uL > uR) and rarefaction (uL < uR) cases.
"""
function riemannInitAna(eq::BurgersEquation, x::Real, t::Real, uL::Real, uR::Real, x0::Real)::Float64
    if t <= 1e-12 # Return initial condition for t=0
        return riemannInit(x, uL, uR, x0)
    end

    if uL > uR
        # --- Shock Wave Case ---
        # Shock speed 's' from Rankine-Hugoniot condition for F(u) = u^2/2
        s = (uL + uR) / 2.0
        shock_position = x0 + s * t
        
        return x < shock_position ? uL : uR

    elseif uL < uR
        # --- Rarefaction Wave Case ---
        # Fan is bounded by characteristics starting from x0 with speeds uL and uR
        x_fan_tail = x0 + uL * t
        x_fan_head = x0 + uR * t

        if x < x_fan_tail
            return uL
        elseif x > x_fan_head
            return uR
        else # Inside the rarefaction fan
            return (x - x0) / t
        end
    else # uL == uR
        # --- Constant State Case ---
        return uL
    end
end


"""
    boxInitAna(x, t, u_background, u_box, x_start, x_end, eq::BurgersEquation)

Provides the analytical solution for Burger's equation for a box-like initial condition.
Handles both the "top-hat" (u_box > u_background) and "well" (u_box < u_background) cases
before the waves interact.
"""
function boxInitAna(eq::BurgersEquation, x::Real, t::Real, u_background::Real, u_box::Real, x_box_start::Real, x_box_end::Real)::Float64
    if t <= 1e-12; return boxInit(x, u_background, u_box, x_box_start, x_box_end); end

    if u_box > u_background # Top-hat case: Rarefaction at left, Shock at right
        s_shock = (u_box + u_background) / 2.0
        x_shock_front = x_box_end + s_shock * t
        x_fan_head = x_box_start + u_box * t
        
        # Check for wave interaction
        if x_fan_head >= x_shock_front
             # Post-interaction solution is more complex. Return NaN as a signal.
             return NaN
        end

        if x < x_box_start + u_background * t
            return u_background
        elseif x < x_fan_head
            return (x - x_box_start) / t
        elseif x < x_shock_front
            return u_box
        else
            return u_background
        end

    elseif u_box < u_background # Well case: Shock at left, Rarefaction at right
        s_shock = (u_background + u_box) / 2.0
        x_shock_front = x_box_start + s_shock * t
        x_fan_tail = x_box_end + u_box * t

        # Check for wave interaction (shock hits rarefaction tail)
        if x_shock_front >= x_fan_tail
            return NaN # Post-interaction solution is more complex.
        end

        if x < x_shock_front
            return u_background
        elseif x < x_fan_tail
            return u_box
        elseif x < x_box_end + u_background * t
            return (x - x_box_end) / t
        else
            return u_background
        end
    else # u_box == u_background
        return u_background
    end
end
# For Burger's Equation
function sineInitAna(eq::BurgersEquation, x::Real, t::Real, a::Real, b_period::Real, c_offset::Real; tol::Real = 1e-10, max_iter::Int = 100)::Float64
    if t <= 1e-12; return sineInit(x, a, b_period, c_offset); end
    u_current::Float64 = sineInit(x, a, b_period, c_offset)
    for _ in 1:max_iter
        u_next = a * sin(2.0 * pi * (x - u_current * t) / b_period) + c_offset
        if abs(u_next - u_current) < tol; return u_next; end
        u_current = u_next
    end
    @warn "sineInitAna: Fixed-point iteration did not converge at x=$x, t=$t."
    return u_current
end
"""
    getInitFunc(eq, name, params, domain_params; bc=:periodic)

Returns a pair of functions: `(init_func_handle, analytic_func_handle)`
based on the equation type and initial condition name.
"""
function getInitFunc(
    eq::ScalarHyperbolicEquation, 
    name::String, 
    params::Tuple, 
    domain_params::NamedTuple; 
    bc::Symbol = :periodic
)
    # --- Select the base initial condition function ---
    base_init_func = if name == "gauss"; gaussInit
                     elseif name == "box"; boxInit
                     elseif name == "sine"; sineInit
                     elseif name == "riemann"; riemannInit
                     else error("Unknown initFunc name: $name"); end
    
    init_func_handle = x -> base_init_func(x, params...)

    # --- Select the analytical solution function ---
    local analytic_func_handle::Function
    if eq isa LinearAdvection
        analytic_func_handle = (x, t) -> linearSolution(x, t, eq, base_init_func, params, domain_params.xmin, domain_params.xmax; bc=bc)
    else
        # For non-linear equations, select the specific analytical solver by name.
        # Julia's multiple dispatch will call the correct version based on the `eq` type.
        ana_solver_func = if name == "sine"; sineInitAna
                          elseif name == "box"; boxInitAna
                          elseif name == "riemann"; riemannInitAna
                          else (eq, x, t, p...) -> NaN; end # Fallback returns NaN
        
        analytic_func_handle = (x, t) -> ana_solver_func(eq, x, t, params...)
    end

    return init_func_handle, analytic_func_handle
end
# function boxInitAna(x::Real, t::Real, u_background::Real, u_box::Real, x_box_start::Real, x_box_end::Real, eq::LinearAdvection{T})::Float64 where T <: Float64
#     if (x > eq.vel * t + x_box_start) & (x < eq.vel * t + x_box_end); return u_box
#     else; return u_background end
# end

# function shockInit2Ana(x::Real, t::Real, xmin, xmax) 
    
#     if t < 10
#         if x<-5; return 0
#         elseif x+5 < t; return t == 0. ?  0 : (x+5)/t
#         elseif x+5>=t && x<= 1/2*t; return 1.
#         else; return 0. end
#     else
#         x_shock_interacting = sqrt(10.0 * t) - 5.0
#         if x < -5.0
#             return 0.0
#         elseif x < x_shock_interacting # Implies -5.0 <= x < sqrt(10*t) - 5 (Rarefaction up to shock)
#             # Ensure u doesn't exceed 1 (though physically it shouldn't if x_s is correct)
#             # The value from rarefaction (x+5)/t is the actual value on the left of the shock
#             val = (x + 5.0) / t
#             # The maximum value the rarefaction can provide before being "cut" by the shock is u_L at the shock
#             # u_L_at_shock = (x_shock_interacting + 5.0) / t = (sqrt(10.0*t) - 5.0 + 5.0) / t = sqrt(10.0/t)
#             # This u_L_at_shock decreases from 1 (at t=10) as t increases.
#             return min(val, sqrt(10.0/t)) # Safeguard, though (x+5)/t should be correct if x < x_s(t)
#                                          # More directly, if x is in the rarefaction fan and to the left of the shock,
#                                          # the value is just (x+5)/t. The shock cuts off the fan.
#         else # x >= x_shock_interacting
#             return 0.0
#         end
#     end
# end
# function shockInit3Ana(x::Real, t::Real)::Float64
#     if t < 0.0
#         error("Time t cannot be negative.")
#     end

#     x_bump_start = -15.0
#     x_bump_end = 10.0
#     u_background = 0.5
#     u_bump = 1.0

#     if t == 0.0
#         # Initial condition
#         if x_bump_start <= x <= x_bump_end
#             return u_bump
#         else
#             return u_background
#         end
#     end

#     # Phase 1 characteristics
#     # Rarefaction from x_bump_start = -15
#     # u_left_of_rarefaction_start = u_background (0.5)
#     # u_right_of_rarefaction_start = u_bump (1.0)
#     x_rarefaction_tail = x_bump_start + u_background * t # -15 + 0.5*t
#     x_rarefaction_head = x_bump_start + u_bump * t      # -15 + 1.0*t

#     # Shock from x_bump_end = 10
#     # u_left_of_shock_start = u_bump (1.0)
#     # u_right_of_shock_start = u_background (0.5)
#     s_initial = (u_bump + u_background) / 2.0 # (1.0 + 0.5) / 2.0 = 0.75
#     x_shock_front_phase1 = x_bump_end + s_initial * t # 10 + 0.75*t

#     # Time of interaction: when head of rarefaction meets the shock path
#     # x_bump_start + u_bump * t_int = x_bump_end + s_initial * t_int
#     # t_int * (u_bump - s_initial) = x_bump_end - x_bump_start
#     # t_int * (1.0 - 0.75) = 10.0 - (-15.0)
#     # t_int * 0.25 = 25.0
#     t_interaction = 100.0

#     if t < t_interaction
#         if x < x_rarefaction_tail
#             return u_background
#         elseif x < x_rarefaction_head # Inside rarefaction fan
#             return (x - x_bump_start) / t # General formula (x-x0)/t + u0 if rarefaction starts from u0 at x0
#                                                         # Here x0 = x_bump_start, u0_char_speed = u_background for tail
#                                                         # u(x,t) = (x - x_bump_start)/t such that u=u_background at x_rarefaction_tail and u=u_bump at x_rarefaction_head
#                                                         # (x_rarefaction_tail - x_bump_start)/t = u_background => ((-15+0.5t) - (-15))/t = 0.5t/t = 0.5. Correct.
#                                                         # (x_rarefaction_head - x_bump_start)/t = u_bump => ((-15+t) - (-15))/t = t/t = 1.0. Correct.
#             return (x - x_bump_start) / t # This gives u in [u_background, u_bump] range
#         elseif x < x_shock_front_phase1 # Plateau
#             return u_bump
#         else # Behind shock
#             return u_background
#         end
#     else # t >= t_interaction
#         x_shock_interacting = x_bump_start + u_background * t + (sqrt(u_bump) - sqrt(u_background))^2 * t + 2 * (sqrt(u_bump) - sqrt(u_background)) * sqrt(t * (x_bump_end - x_bump_start - (u_bump+u_background)/2 * t_interaction_check_again))
#         # The previous derivation: xs(t) = -15 + 0.5*t + 5*sqrt(t) is correct.
#         x_shock_interacting_val = -15.0 + 0.5 * t + 5.0 * sqrt(t)


#         if x < x_rarefaction_tail # x < -15 + 0.5t
#             return u_background
#         elseif x < x_shock_interacting_val # Rarefaction up to the shock
#                                           # Value is (x - x_bump_start)/t
#             return (x - x_bump_start) / t
#         else # Behind shock
#             return u_background
#         end
#     end
# end

# # Helper function to map a value back to the periodic domain [xmin, xmax)
# function periodic_map(y::Real, xmin::Real, xmax::Real)
#     L = xmax - xmin
#     # Ensures the result is in [0, L) then shifts by xmin
#     return xmin + mod(y - xmin, L)
# end

# # Function to numerically solve the implicit analytical solution for smooth ICs with periodicity
# function solve_implicit_periodic(x::Real, t::Real, xmin::Real, xmax::Real, f::Function; tol::Real = 1e-9, max_iter::Int = 100)
#     if t == 0.0
#         return f(x)
#     end

#     u_old = f(x) # Initial guess

#     for _ in 1:max_iter
#         # Calculate the characteristic foot at t=0, mapped to the periodic domain
#         xi = periodic_map(x - u_old * t, xmin, xmax)
#         u_new = f(xi)
#         if abs(u_new - u_old) < tol
#             return u_new
#         end
#         u_old = u_new
#     end
#     # If iteration doesn't converge, return the last value and warn
#     @warn "Fixed-point iteration for analytical solution did not converge at x=$x, t=$t. Max iterations reached."
#     return u_old
# end

# # Analytical solution for smoothInit1 with periodicity (iterative evaluation)
# function smoothInit1AnaPeriodic(x::Real, t::Real, xmin::Real, xmax::Real, eq::BurgersEquation)
#     return solve_implicit_periodic(x, t, xmin, xmax, smoothInit1)
# end



# # Analytical solution for smoothInit2 with periodicity (iterative evaluation)
# function smoothInit2Ana(x::Real, t::Real, xmin::Real, xmax::Real, eq::BurgersEquation)
#     return solve_implicit_periodic(x, t, xmin, xmax, smoothInit2)
# end

# function sineInitAna(eq::LinearAdvection{T}, x::Real, t::Real, a::Real, b::Real, c::Real) where T <: Float64
#     return a*sin(2*pi*(x-eq.vel*t)/b) + c
# end

# # Ensure BurgersEquation is defined, e.g., from your ScalarHyperbolicEquations module
# # struct BurgersEquation <: NonLinearScalarHyperbolicEquation end # Minimal definition if needed here
# # using ..Meshfree4ScalarEq.ScalarHyperbolicEquations # Or your actual path

# function sineInitAna(eq::BurgersEquation, 
#                                      x::Real, t::Real, 
#                                      a::Real, b_period::Real, c_offset::Real;
#                                      tol::Real = 1e-10, max_iter::Int = 1000)::Float64
    
#     if t < 0.0
#         error("Time t cannot be negative.")
#     end

#     initial_condition_func = (x0::Real) -> a * sin(2.0 * pi * x0 / b_period) + c_offset

#     if abs(t) < 1e-14 # Effectively t == 0.0
#         return initial_condition_func(x)
#     end

#     # Calculate theoretical shock formation time
#     t_shock_formation = Inf
#     if abs(a) > 1e-14 && abs(b_period) > 1e-14
#         min_derivative_val = -abs(a * (2.0 * pi / b_period))
#         if abs(min_derivative_val) > 1e-14
#             t_shock_formation = -1.0 / min_derivative_val
#         end
#     end

#     if t >= t_shock_formation && abs(a) > 1e-14 # Warn if t is beyond expected smooth regime
#         # This analytical solution method (simple fixed point for smooth u) is not valid after shock.
#         # Depending on the exact x, it might still give a value from one of the branches.
#         # For strictness, one could error or return NaN.
#         # However, for plotting/comparison, letting it attempt and warn is common.
#         # @warn "Attempting analytical solution at t=$t which may be at or after shock time t_sh=$t_shock_formation."
#     end

#     u_current::Float64 = initial_condition_func(x) # Initial guess for u(x,t) is u(x,0)
#     u_next::Float64 = 0.0

#     for iter_count in 1:max_iter
#         x_characteristic_foot = x - u_current * t # x0 on infinite domain
#         u_next = initial_condition_func(x_characteristic_foot)
        
#         if abs(u_next - u_current) < tol
#             return u_next
#         end
#         u_current = u_next
#     end

#     @warn "sineInitAna_infinite_domain: Fixed-point iteration did not converge at x=$x, t=$t after $max_iter iterations. Last diff: $(abs(u_next - u_current)). Approx. shock time: $t_shock_formation. Returning last iterate."
#     return u_current 
# end




# -------------------------------------------------


"""
    runBurgersSimulation_for_IPlotPDESols(params::ParamDictType)

Runs a single 1D Burgers' equation simulation based on parameters defined
in the input dictionary and returns results compatible with IPlotPDESols.
Requires all necessary parameters to be present in the input `params` dict.
Calculates `dt` based on `CFL` and the Burgers equation properties.

# Arguments
- `params::ParamDictType`: Dictionary containing ALL necessary simulation parameters.
                           Expected keys: "method" (String matching names from burgers.txt),
                           "initFunc" (String), "N", "xmin", "xmax", "regular", "tmax",
                           "CFL", "saveFreq", "interpAlpha", "interpRangeFactor",
                           "randomness_factor" (if regular=false).

# Returns
- `AbstractSimData`: The simulation result object (e.g., SimData1D), or `nothing` on error.
"""
function RunSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    println("\n--- Running Burgers Simulation via IPlotPDESols Interface ---")
    run_params = copy(params) # Work on a copy to store derived values

    # --- Manage RNG State for Reproducibility ---
    local rng_state_backup
    #rng_defined = @isdefined(Meshfree4ScalarEq.rng, :rng) && isa(Meshfree4ScalarEq.rng, Random.AbstractRNG)
    rng_state_backup = copy(Meshfree4ScalarEq.rng);
    # -----------------------------------------

    local sim_data_result = nothing # Ensure defined outside try

    try
        # --- Extract REQUIRED Parameters (Direct Access) ---
        # --- Extract REQUIRED Parameters (Direct Access) ---
        tmax::Float64 = run_params["tmax"]
        N_particles::Int = run_params["N"]
        bc::Symbol = run_params["bc"]
        xmin::Float64 = run_params["xmin"]
        xmax::Float64 = run_params["xmax"]
        initFunc_name::String = run_params["init_func"]
        cfl = get(run_params, "CFL", nothing)
        dt = get(run_params, "dt", nothing)
        save_freq::Int = run_params["save_frequency"]
        interp_alpha::Float64 = run_params["interp_alpha"]
        interp_range_factor::Float64 = run_params["interp_range"]
        randomness_factor::Float64 = run_params["randomness_factor"] # Required
        order::Int = run_params["order"] # <<< Treat order as required
        timestepper_name = run_params["timestepper"] # Timestepper needed
        eq_name = run_params["PDE"]

        # --- Extract METHOD Parameters (Use `get` with sensible defaults) ---
        mood_name = get(run_params, "MOOD", nothing)
        delta_relax = get(run_params, "delta_relax", nothing)
        println(delta_relax)
        main_grad_name = get(run_params, "main_gradient", nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing) # Default fallback = 1st order Upwind
        main_flux_name = get(run_params, "main_flux", nothing)
        fallback_flux_name = get(run_params, "fallback_flux", nothing)
        switch_tol = get(run_params, "switch_tol", nothing)
        init_params = (get(run_params, "init_params", nothing))
        relax_vel = get(run_params, "relax_velocities", nothing)
        relax_method = get(run_params, "relax_method", false)
        relax_eps = get(run_params, "relax_epsilon", nothing)
        eq_params = get(run_params, "PDE_params", nothing)
        lim = get(run_params, "limiter", nothing)

        # --- Derive regularity ---
        regular::Bool = (randomness_factor == 0.0)
        N_ghost::Int64 = bc == :periodic ? 0 : convert(Int64, ceil(interp_range_factor))

        N = N_particles + 2*N_ghost

    
        println("  TimeStepper = $timestepper_name, Main Gradient = $main_grad_name ($order), Main Flux = $main_flux_name")
        println("  Fallback = $fallback_grad_name / $fallback_flux_name, MOOD = $mood_name (deltaRelax=$delta_relax)")
        println("  IC = $initFunc_name, N = $N_particles, Regular = $regular (randFactor=$randomness_factor), Order = $order")
        println("  tmax = $tmax, CFL = $cfl")
        println("----------------------------------------")

        # --- Grid Creation ---
        dx_nominal = (xmax - xmin) / N_particles
        local particleGrid

        randomness = randomness_factor * dx_nominal
        particleGrid = ParticleGrid1D(xmin, xmax, N_particles, N_ghost, bc; randomness = randomness)
        determineVolumes!(particleGrid)
        # --- Calculate Dependent Parameters ---
        interp_range = interp_range_factor * particleGrid.dx
        # Calculate dt based on CFL using the *correct* equation (Burgers)
        # --- Equation ---
        if eq_name == "burgers"
            eq = BurgersEquation() # Instantiate Burgers' equation object
        elseif eq_name == "linear"
            eq = LinearAdvection(eq_params...)
        else 
            error("Requested PDE not implemented!")
        end
        F = u -> flux(eq, u)
        # This assumes getTimeStep can handle BurgersEquation appropriately
        if !isnothing(cfl)
            eqLin = eq_name == "linear" ? eq : LinearAdvection(1.0)
            dt = cfl * getTimeStep(particleGrid, eqLin, interp_alpha, interp_range)
        elseif isnothing(dt)
            error("The time step has to be given directly via the dt-key or via the CFL fraction using the CFL-key!")
        end

        # --- Build Method Components (Ensure types/constructors are accessible) ---
        local mood_fun
        if mood_name == "U1"; mood_fun = MOODu1(deltaRelax = delta_relax)
        elseif mood_name == "U2"; mood_fun = MOODu2(deltaRelax = delta_relax)
        elseif mood_name == "LoubertU2"; mood_fun = MOODLoubertU2(deltaRelax = delta_relax)
        elseif mood_name == "none"; mood_fun = NoMOOD()
        elseif mood_name == "only"; mood_fun = OnlyMOOD()
        elseif mood_name == "firstNone"; mood_fun = FirstStageNoMOOD()
        elseif mood_name == "alt"; mood_fun = MOODAlt(delta_relax)
        elseif !isnothing(mood_name); error("MOOD '$mood_name' not recognized") end

        local limiter
        if !isnothing(lim)
            @assert (main_grad_name == "MUSCL") "Slope limiter only supported for MUSCL-schemes!"
            @assert (order == 2) "Only linear reconstruction supported at the moment!"
        end 
        if lim == "minmod"
            limiter = MinmodLimiter()
        elseif lim == "superbee"
            limiter = SuperbeeLimiter()
        elseif lim == "VK"
            limiter = VenkatakrishnanLimiter()
        elseif lim == "BJ"
            limiter = BarthJespersenLimiter()
        elseif !isnothing(lim); error("Limiter '$lim' not recognized") end
        local MainFlux
        if main_flux_name == "LW"; MainFlux = LaxWendroffFlux()
        elseif main_flux_name == "Rusanov"; MainFlux = RusanovFlux()
        elseif main_flux_name == "Upwind"; MainFlux = UpwindFlux()
        elseif !isnothing(main_flux_name); error("Requested Main Flux is not implemented!") end

        local FallbackFlux
        if fallback_flux_name == "LW"; FallbackFlux = LaxWendroffFlux()
        elseif fallback_flux_name == "Rusanov"; FallbackFlux = RusanovFlux()
        elseif fallback_flux_name == "Upwind"; FallbackFlux = UpwindFlux()
        elseif !isnothing(fallback_flux_name); error("Requested Fallback Flux is not implemented!") end

        local MainGrad
        if main_grad_name == "MUSCL"; MainGrad = isnothing(lim) ? MUSCL(order-1; numericalFlux = MainFlux) : MUSCLlimited(1; numericalFlux = MainFlux, limiter = limiter)
        #elseif main_grad_name == "MUSCLlimit"; MainGrad = MUSCLlimited(1; numericalFlux = MainFlux)
        elseif main_grad_name == "Upwind"; MainGrad = UpwindGradient(order; numericalFlux = MainFlux)
        elseif main_grad_name == "WENO"; error("WENO not implemented for non-linear case.")
        elseif !isnothing(main_grad_name); error("Requested Main GradientInterpolator not implemented!") end

        local FallbackGrad 
        if fallback_grad_name == "MUSCL"; error("No 1st order MUSCL method defined") #FallbackGrad = MUSCL(1; numericalFlux = FallbackFlux)
        elseif fallback_grad_name == "Upwind"; FallbackGrad = UpwindGradient(1; numericalFlux = FallbackFlux)
        elseif fallback_grad_name == "WENO"; error("WENO not implemented for non-linear case.")
        elseif isnothing(fallback_grad_name); FallbackGrad = nothing 
        else; error("Requested Fallback GradientInterpolator not implemented!") end
        # --- Time Stepper Selection ---
        local method
        if timestepper_name == "RalstonRK2"; method = RalstonRK2(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun)
        elseif timestepper_name == "EulerUpwind"; method = EulerUpwind(N; gradientInterpolator = MainGrad) # Assumes EulerUpwind ignores fallback/mood args if passed
        elseif timestepper_name == "RK3"; method = RK3(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun)
        elseif timestepper_name == "RK4"; method = RK4(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun)
        elseif timestepper_name == "LF"; method = LaxFriedrich(N)
        elseif timestepper_name == "Classic"; method = ClassicalTimeStepper(N, MainFlux)
        elseif timestepper_name == "Upwind"; method = Upwind(N)
        elseif timestepper_name == "Analytic"; method = nothing 
        elseif timestepper_name == "RalstonRK2SmoothSwitch"; method = RalstonRK2SmoothSwitch2(MainGrad, N; fallbackInterpolator = FallbackGrad, mood = mood_fun, tol = switch_tol)
        elseif relax_method @info "Relaxation Method Detected!"
        else; error("Unknown TimeStepper name: '$timestepper_name'"); end



        # --- Initial Condition ---
        local init_func_handle::Function
        init_func_handle, analytic_func = getInitFunc(eq, initFunc_name, init_params,(xmin = xmin, xmax = xmax); bc = bc)
        # if initFunc_name == "sine"; init_func_handle(x) = sineInit(x, init_params...); analytic_func(x,t) = sineInitAna(eq, x, t, init_params...)
        # elseif initFunc_name == "gauss"
        #     init_func_handle = x -> gaussInit(x, init_params...)
        #     if eq_name == "linear" 
        #         analytic_func = (x,t) -> linearSolution(x, t, eq, gaussInit, init_params, xmin, xmax)
        #     else 
        #         error("Non-linear analytic function not implemented yet!")
        #     end
        # # elseif initFunc_name == "smoothInit2"; init_func_handle = smoothInit2
        # # elseif initFunc_name == "shockInit1"; init_func_handle = shockInit1
        # # elseif initFunc_name == "shockInit2"; init_func_handle = shockInit2; analytic_func = (x,t) -> shockInit2Ana(x,t,xmin,xmax)
        # # elseif initFunc_name == "shockInit3"; init_func_handle = shockInit3; analytic_func = shockInit3Ana
        # elseif initFunc_name == "box"
        #     @assert typeof(init_params) <: Tuple{Real,Real,Real,Real} "Box Init needs a tuple of 4 real numbers as parameters!"
        #     u_background, u_box, box_start, box_end = init_params 
        #     @assert u_box >= u_background "Only top hat supported atm!"
        #     @assert box_end > box_start "The end of the box has to be larger than the start!"
        #     init_func_handle = x -> boxInit(x, init_params...)
        #     analytic_func = (x,t) -> boxInitAna(x,t,u_background,u_box,box_start,box_end,eq)
        # else; error("Unknown initFunc name: $initFunc_name"); end
        # if eq_name == "linear" 
        #         analytic_func = (x,t) -> linearSolution(x, t, eq, init_func_handle, init_params, xmin, xmax)
        # end
        setInitialConditions!(particleGrid, init_func_handle)
            # Create SimSettings object
        settings = SimSetting(  tmax=tmax,
                                dt=dt,
                                interpRange=interp_range,
                                interpAlpha= interp_alpha,
                                saveDir="/", 
                                saveFreq=save_freq,
                                organiseFiles = false)

        # --- Call the NEW Time Integrator ---
        println("Starting time integration (Burgers)...")
        if relax_method
            @assert !isnothing(relax_vel) "The relaxation method needs velocities to create the linear system!"
            eqs = [LinearAdvection(a) for a = relax_vel]
            pgs = [deepcopy(particleGrid) for _ = relax_vel]
            M = [(rho -> 1/2 * (rho + F(rho)/a)) for a = relax_vel]
            for (pg_idx,pg) = enumerate(pgs)
                for particle = pg.grid
                    particle.rho = M[pg_idx](particle.rho)
                end
            end
            source_term = RelaxationSourceTerm1D(M, relax_eps)
            implicit_solver = LinearizedRelaxationImplicitSolver()
            if timestepper_name =="ARS2"
                system_method = ARS2IMEX(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            elseif timestepper_name == "ARS233"
                system_method = ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            elseif timestepper_name == "PRSSP3"
                system_method = PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            elseif timestepper_name == "ARS222"
                system_method = ARS222(MainGrad,FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            elseif timestepper_name == "SSP2332"
                system_method = SSP2332(MainGrad,FallbackGrad, mood_fun, implicit_solver, source_term, N, length(relax_vel))
            else
                system_method = RelaxationStepper(method, N, M; epsilon = relax_eps)
            end
            elapsed_time, sys_xs, sys_us, ts = mainTimeIntegrator2!(system_method, eqs, pgs, settings)
            us = [vec(sum(sys_u, dims=2)) for sys_u = sys_us]
            xs = [sys_x[:,1] for sys_x = sys_xs]
            #println(typeof(us), typeof(xs))
        elseif !isnothing(method)
            elapsed_time, xs, us, ts = mainTimeIntegrator2!(method, eq, particleGrid, settings)
        else
            elapsed_time = 0.
            
            ts = collect(0:dt*save_freq:tmax)
            println(ts[end])
            if ts[end] != tmax
                push!(ts, tmax)
            end
            println(ts[end])
            xs = [collect(range(xmin, xmax, N_particles)) for _ = ts]
            us =  Vector{Vector{Float64}}(undef, 0)
            for (i,t) in enumerate(ts)
                u_tmp = map(x -> analytic_func(x,t), xs[i])
                push!(us, u_tmp)
            end
        end
        #println(typeof(xs), typeof(us), typeof(ts))
        println("Time integration finished in $(round(elapsed_time, digits=2)) seconds.")

        sim_data_result = createSimData(xs, us, ts, run_params)
        # --- Post-processing ---
        if !isnothing(sim_data_result)
             # Add metadata to stats dictionary
             calculateAllStats!(sim_data_result, analytic_func, (xmin = xmin, xmax = xmax), N_particles; quad_tol = 10e-14, dierckx_k = 3)
             if hasproperty(sim_data_result, :stats) && isa(sim_data_result.stats, Dict)
                 sim_data_result.stats["time"] = elapsed_time
                #  sim_data_result.stats["dx_nominal"] = dx_nominal
                #  sim_data_result.stats["dt_calculated"] = dt
                #  sim_data_result.stats["interp_range_calculated"] = interp_range
                 #sim_data_result.stats["method_type"] = string(typeof(method))
             else
                 @warn "Could not add metadata to stats field in SimData."
             end
             # Ensure params field matches original input params for hashing consistency
             # If mainTimeIntegrator! modified run_params_for_integrator, copy original params back
             # sim_data_result.params = params # Check if necessary based on mainTimeIntegrator! behavior
        end
        println("Length of t",length(ts))
        return sim_data_result # Return the SimData object

    catch e
        rethrow(e)
         if isa(e, KeyError); @error "Missing required parameter!" key=e.key params=params
         else; @error "Error during Burgers simulation setup or execution!" params=params exception=(e, catch_backtrace()); end
         return nothing # Return nothing on error
    finally
        # Restore RNG state regardless of success/failure
        copy!(Meshfree4ScalarEq.rng, rng_state_backup)
    end
end
SEED_value = (:const, Meshfree4ScalarEq.SEED)
# Example SimulationConfig for Burgers
sim_config_burgers = SimulationConfig(
    RunSimulation, # Use the new runner

    ParamDict(
        "tmax" => 30, "N" => 200, "xmin" => -5.0, "xmax" => 18.0,
        "CFL" => .2, "save_frequency" => 10, "interp_alpha" => 1.0,
        "interp_range" => 1.5,
        "init_func" => "box",
        "init_params" => (0., 1., -4., -2.), #(1., 0., 1.)
        "randomness_factor" => 0., # Provide default needed when regular=false
        "SEED" => SEED_value, 
        "timestepper" => "Classic", "bc" => :outflow,
        "order" => 1, "PDE" => "linear", "PDE_params" => (.5,)
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
            "delta_relax" => 1.,
            "order" => 2
        ),
        "RK2MUSCL2MOOD(LoubertU2)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "LoubertU2",
            "delta_relax" => 1.,
            "order" => 2
        ),
        "RK2MUSCL2MOOD(U1)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U1",
            "delta_relax" => 1.,
            "order" => 2
        ),
        "RK2MUSCL2MOOD" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 2
        ),
        "RK2MUSCL2" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 2
        ),
        "RK2MUSCL2MOOD(relax)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 10000.,
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
        "EulerUpwind" => ParamDict(
            "timestepper" => "EulerUpwind",
            "main_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "order" => 1
        ),
        "Analytic Solution" => ParamDict(
            "timestepper" => "Analytic",
            "randomness_factor" => (:const,0.),
            "N" => (:const, 1000), 
            "save_frequency" => 50
             # No randomness_factor needed when regular=true
        ),
        "LLF(uniform grid)" => ParamDict(
            "randomness_factor" => (:const,0.),
            "main_flux" => "Rusanov"
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
    "RK2MUSCL2" #, "Relax Method 2", "Relax Method 3rd order","Classic","SlopeLimiter","SmoothSwitching","Regular MOOD", "OnlyFallback"]
);

# Pass this config to your IPlotPDESols functions
show1DSolutionFig(sim_config_burgers; ui_options = :default);
showDynamicDependence(sim_config_burgers; ui_options = :publication)
#calculateConvergenceData(sim_config_burgers, "N", 10. .^(1:.25:2.5); force_int_param = true)
#showConvergencePlot(sim_config_burgers, "N", 10. .^(1:.25:2.5); force_int_param = true, initial_calc = true)
#showConvergencePlot(sim_config_burgers, "delta_relax", 0:10^-101:10^-100; force_int_param = false, initial_calc = false)
#showConvergencePlot(sim_config_burgers, "switch_tol", 10. .^(-5:.1:-2); force_int_param = false, initial_calc = false)