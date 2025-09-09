abstract type HyperbolicSystem <: HyperbolicPDE end

# --- 1D Euler Equations ---

struct Euler1D <: HyperbolicSystem end

const GAS_GAMMA_EULER_1D = 1.4

"""
    pressure_from_euler_conserved(rho, m, E)

Calculates pressure from conserved variables for the 1D Euler equations.
"""
function pressure_from_euler_conserved(rho::Float64, m::Float64, E::Float64)::Float64
    if rho < 1e-9; return 1e-9; end
    pressure = (GAS_GAMMA_EULER_1D - 1.0) * (E - 0.5 * m^2 / rho)
    return max(pressure, 1e-9)
end

"""
    flux(eq::Euler1D, rho, m, E)

Returns the physical flux vector for the 1D Euler equations.
"""
function flux(eq::Euler1D, rho::Float64, m::Float64, E::Float64)::NTuple{3, Float64}
    if rho < 1e-9; return (0.0, pressure_from_euler_conserved(1e-9,0.0,0.0), 0.0); end
    ux = m / rho
    p = pressure_from_euler_conserved(rho, m, E)
    return (m, m * ux + p, (E + p) * ux)
end


# --- 2D Euler Equations ---

struct Euler2D <: HyperbolicSystem end

const GAS_GAMMA_EULER_2D = 1.4

"""
    pressure_from_euler_conserved(rho, mx, my, E)

Calculates pressure from conserved variables for the 2D Euler equations.
"""
function pressure_from_euler_conserved(rho::Float64, mx::Float64, my::Float64, E::Float64)::Float64
    if rho < 1e-9; return 1e-9; end
    pressure = (GAS_GAMMA_EULER_2D - 1.0) * (E - 0.5 * (mx^2 + my^2) / rho)
    return max(pressure, 1e-9)
end
function pressure_from_euler_conserved(U::NTuple{4,Float64})::Float64
    rho, mx, my, E = U
    if rho < 1e-9; return 1e-9; end
    pressure = (GAS_GAMMA_EULER_2D - 1.0) * (E - 0.5 * (mx^2 + my^2) / rho)
    return max(pressure, 1e-9)
end

"""
    flux(eq::Euler2D, rho, mx, my, E)

Returns the two physical flux vectors (F, G) for the 2D Euler equations.
"""
function flux(eq::Euler2D, rho::Float64, mx::Float64, my::Float64, E::Float64)::NTuple{2, NTuple{4, Float64}}
    if rho < 1e-9
        p_fallback = pressure_from_euler_conserved(1e-9, 0.0, 0.0, 0.0)
        return ((0.0, p_fallback, 0.0, 0.0), (0.0, 0.0, p_fallback, 0.0))
    end
    p = pressure_from_euler_conserved(rho, mx, my, E)
    ux = mx / rho
    uy = my / rho
    F = (rho * ux, rho * ux^2 + p, rho * ux * uy, (E + p) * ux)
    G = (rho * uy, rho * ux * uy, rho * uy^2 + p, (E + p) * uy)
    return F, G
end

function flux(eq::Euler2D, U::NTuple{4,Float64})::NTuple{2, NTuple{4, Float64}}
    rho, mx, my, E = U
    if rho < 1e-9
        p_fallback = pressure_from_euler_conserved(1e-9, 0.0, 0.0, 0.0)
        return ((0.0, p_fallback, 0.0, 0.0), (0.0, 0.0, p_fallback, 0.0))
    end
    p = pressure_from_euler_conserved(U)
    ux = mx / rho
    uy = my / rho
    F = (rho * ux, rho * ux^2 + p, rho * ux * uy, (E + p) * ux)
    G = (rho * uy, rho * ux * uy, rho * uy^2 + p, (E + p) * uy)
    return F, G
end
