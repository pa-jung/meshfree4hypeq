# Suggested new file: src/InitialConditions.jl

module InitialConditions

# Import necessary types from your main module. Adjust the path as needed.
using ..ScalarHyperbolicEquations 
using ..ParticleGrids

export InitialCondition, SmoothInitialCondition, ShockInitialCondition, 
       Gauss, Box, Sine, Riemann, 
       getInitialCondition, get_discontinuity_points

# --- 1. Abstract Type Hierarchy ---
abstract type InitialCondition end
abstract type SmoothInitialCondition <: InitialCondition end
abstract type ShockInitialCondition <: InitialCondition end


# --- 2. Concrete Structs and Functors for t=0 ---

# --- GAUSS (Smooth) ---
struct Gauss <: SmoothInitialCondition
    a::Float64      # amplitude
    b::Float64      # mean (center)
    width::Float64
end
# Functor for initial condition at t=0
(ic::Gauss)(x::Real) = ic.a * exp(-((x - ic.b) / ic.width)^2)
(ic::Gauss)(x::Real, y::Real) = error("2D Gauss IC not implemented.")


# --- SINE (Smooth) ---
struct Sine <: SmoothInitialCondition
    a::Float64      # amplitude
    b_period::Float64
    c_offset::Float64
end
(ic::Sine)(x::Real) = ic.a * sin(2.0 * pi * x / ic.b_period) + ic.c_offset
(ic::Sine)(x::Real, y::Real) = error("2D Sine IC not implemented.")


# --- BOX (Shock) ---
struct Box <: ShockInitialCondition
    u_background::Float64
    u_box::Float64
    x_start::Float64
    x_end::Float64
end
(ic::Box)(x::Real) = ic.x_start <= x <= ic.x_end ? ic.u_box : ic.u_background
(ic::Box)(x::Real, y::Real) = error("2D Box IC not implemented.")


# --- RIEMANN (Shock) ---
struct Riemann <: ShockInitialCondition
    uL::Float64
    uR::Float64
    x0::Float64
end
(ic::Riemann)(x::Real) = x < ic.x0 ? ic.uL : ic.uR
(ic::Riemann)(x::Real, y::Real) = error("2D Riemann IC not implemented.")


# --- 3. Analytical Solution Functors (t>0) using Multiple Dispatch ---

# --- For Linear Advection (General Solution) ---
function (ic::InitialCondition)(x::Real, t::Real, eq::LinearAdvection, pg::ParticleGrid1D)
    x0 = x - eq.vel * t
    if pg.bc == :periodic
        domain_length = pg.xmax - pg.xmin
        x0_mapped = pg.xmin + mod(x0 - pg.xmin, domain_length)
        return ic(x0_mapped)
    else # :fixed or :outflow (infinite domain assumption)
        return ic(x0)
    end
end

(ic::InitialCondition)(x::Real, y::Real, t::Real, eq::LinearAdvection, pg::ParticleGrid2D) = error("2D analytical linear advection not implemented.")


# --- For Burger's Equation (Specific to each IC Type) ---
function (ic::Sine)(x::Real, t::Real, eq::BurgersEquation, pg::ParticleGrid1D; tol::Real = 1e-10, max_iter::Int = 100)
    if t <= 1e-12; return ic(x); end
    u_current = ic(x)
    for _ in 1:max_iter
        u_next = ic.a * sin(2.0 * pi * (x - u_current * t) / ic.b_period) + ic.c_offset
        if abs(u_next - u_current) < tol; return u_next; end
        u_current = u_next
    end
    @warn "sineInitAna: Fixed-point iteration did not converge at x=$x, t=$t."
    return u_current
end

function (ic::Box)(x::Real, t::Real, eq::BurgersEquation, pg::ParticleGrid1D)
    # --- Pre-computation and edge cases ---
    if t <= 1e-12; return ic(x); end
    if abs(ic.u_box - ic.u_background) < 1e-12; return ic.u_background; end

    if ic.u_box > ic.u_background
        # --- Top-hat case: Rarefaction at left, Shock at right ---
        
        # Time of interaction: when the head of the rarefaction fan catches the shock
        # This occurs when the plateau of u_box disappears.
        delta_u = ic.u_box - ic.u_background
        t_interaction = 2.0 * (ic.x_end - ic.x_start) / delta_u
        
        if t < t_interaction
            # --- Phase 1: Waves evolve independently ---
            s_shock = (ic.u_box + ic.u_background) / 2.0
            x_shock_front = ic.x_end + s_shock * t
            x_fan_head = ic.x_start + ic.u_box * t
            
            if x < ic.x_start + ic.u_background * t
                return ic.u_background
            elseif x < x_fan_head
                # Inside rarefaction fan
                return (x - ic.x_start) / t
            elseif x < x_shock_front
                # Plateau region
                return ic.u_box
            else 
                # Behind shock
                return ic.u_background
            end
        else
            # --- Phase 2: Shock has merged with rarefaction fan ---
            # The shock path is now x_s(t) = x_start + u_background*t + C*sqrt(t)
            C = sqrt(2.0 * (ic.x_end - ic.x_start) * delta_u)
            x_shock_interacting = ic.x_start + ic.u_background * t + C * sqrt(t)

            if x < ic.x_start + ic.u_background * t
                return ic.u_background
            elseif x < x_shock_interacting
                # Inside rarefaction fan, up to the interacting shock
                return (x - ic.x_start) / t
            else 
                # Behind the interacting shock
                return ic.u_background
            end
        end

    else # u_box < u_background
        # --- Well case: Shock at left, Rarefaction at right ---

        # Time of interaction: when the shock front catches the tail of the rarefaction fan
        delta_u = ic.u_background - ic.u_box
        t_interaction = 2.0 * (ic.x_end - ic.x_start) / delta_u

        if t < t_interaction
            # --- Phase 1: Waves evolve independently ---
            s_shock = (ic.u_background + ic.u_box) / 2.0
            x_shock_front = ic.x_start + s_shock * t
            x_fan_tail = ic.x_end + ic.u_box * t
            
            if x < x_shock_front
                return ic.u_background
            elseif x < x_fan_tail
                return ic.u_box
            elseif x < ic.x_end + ic.u_background * t
                # Inside rarefaction fan
                return (x - ic.x_end) / t
            else
                return ic.u_background
            end
        else
            # --- Phase 2: Shock has entered the rarefaction fan ---
            # The shock path is now x_s(t) = x_end + u_background*t - C*sqrt(t)
            C = sqrt(2.0 * (ic.x_end - ic.x_start) * delta_u)
            x_shock_interacting = ic.x_end + ic.u_background * t - C * sqrt(t)

            if x < x_shock_interacting
                return ic.u_background
            elseif x < ic.x_end + ic.u_background * t
                # Inside rarefaction fan, to the right of the interacting shock
                return (x - ic.x_end) / t
            else
                return ic.u_background
            end
        end
    end
end

function (ic::Riemann)(x::Real, t::Real, eq::BurgersEquation, pg::ParticleGrid1D)
    if t <= 1e-12; return ic(x); end
    if ic.uL > ic.uR # Shock
        s = (ic.uL + ic.uR) / 2.0
        shock_pos = ic.x0 + s * t
        return x < shock_pos ? ic.uL : ic.uR
    else # Rarefaction
        x_fan_tail = ic.x0 + ic.uL * t
        x_fan_head = ic.x0 + ic.uR * t
        if x < x_fan_tail; return ic.uL;
        elseif x > x_fan_head; return ic.uR;
        else return (x - ic.x0) / t; end
    end
end

# Fallback for ICs without a specific analytical solution for Burger's
(ic::Gauss)(x::Real, t::Real, eq::BurgersEquation, pg::ParticleGrid1D) = NaN


# --- 4. Factory Function ---
"""
    getInitialCondition(name::String, params::Tuple)

Factory function that returns an instance of the correct InitialCondition struct.
"""
function getInitialCondition(name::String, params::Tuple)::InitialCondition
    if name == "gauss"; return Gauss(params...);
    elseif name == "box"; return Box(params...);
    elseif name == "sine"; return Sine(params...);
    elseif name == "riemann"; return Riemann(params...);
    else error("Unknown initFunc name: $name"); end
end


# --- 5. Discontinuity Finder ---
"""
    get_discontinuity_points(ic, eq, t, pg)

Calculates the current positions of any discontinuities for QuadGK.
Uses multiple dispatch on the initial condition type and equation type.
"""
# Default for smooth ICs with linear advection -> no discontinuities
get_discontinuity_points(ic::SmoothInitialCondition, eq::LinearAdvection, t::Real, pg::ParticleGrid) = Float64[]

# For shock ICs with linear advection -> track the initial jumps
function get_discontinuity_points(ic::ShockInitialCondition, eq::LinearAdvection, t::Real, pg::ParticleGrid1D)
    points = Float64[]
    xmin, xmax = pg.xmin, pg.xmax
    domain_length = xmax - xmin
    
    initial_points = if ic isa Box; [ic.x_start, ic.x_end]; else [ic.x0]; end
    
    for pt in initial_points
        advected_pos = pt + eq.vel * t
        if pg.bc == :periodic
            advected_pos = xmin + mod(advected_pos - xmin, domain_length)
        end
        push!(points, advected_pos)
    end
    return unique(sort(points))
end

# For Riemann with Burger's -> track shock or rarefaction edges
function get_discontinuity_points(ic::Riemann, eq::BurgersEquation, t::Real, pg::ParticleGrid1D)
    if ic.uL > ic.uR # Shock
        s = (ic.uL + ic.uR) / 2.0
        return [ic.x0 + s * t]
    else # Rarefaction
        return [ic.x0 + ic.uL * t, ic.x0 + ic.uR * t]
    end
end

# For Box with Burger's -> track shock and rarefaction edges
function get_discontinuity_points(ic::Box, eq::BurgersEquation, t::Real, pg::ParticleGrid1D)
    s_shock = (ic.u_box + ic.u_background) / 2.0
    if ic.u_box > ic.u_background # Top-hat
        res = [ic.x_start + ic.u_background * t]
        rare_pos = ic.x_start + ic.u_box * t
        shock_pos = ic.x_end + s_shock * t
        if rare_pos < shock_pos
            push!(res, rare_pos)
        end
        push!(res, shock_pos)
        return res
    else # Well
        return [ic.x_start + s_shock * t, ic.x_end + ic.u_box * t, ic.x_end + ic.u_background * t]
    end
end

# For Sine with Burger's -> track the characteristic from the point of steepest descent
function get_discontinuity_points(ic::Sine, eq::BurgersEquation, t::Real, pg::ParticleGrid1D)
    # Shock forms at the point with the most negative slope.
    # For a*sin(2pi*x/b), this is at x = b/2 (if a>0) or x=0 (if a<0)
    x_break = ic.a > 0 ? ic.b_period / 2.0 : 0.0
    u_at_break = ic(x_break)
    return [x_break + u_at_break * t]
end

# For Gauss with Burger's -> track the characteristic from the inflection point
function get_discontinuity_points(ic::Gauss, eq::BurgersEquation, t::Real, pg::ParticleGrid1D)
    # Shock forms at the point with the most negative slope (an inflection point).
    x_break = ic.a > 0 ? ic.b + ic.width / sqrt(2.0) : ic.b - ic.width / sqrt(2.0)
    u_at_break = ic(x_break)
    return [x_break + u_at_break * t]
end

# 2D Placeholders
get_discontinuity_points(ic::InitialCondition, eq::ScalarHyperbolicEquation, t::Real, pg::ParticleGrid2D) = error("2D discontinuity tracking not implemented.")


end # module InitialConditions