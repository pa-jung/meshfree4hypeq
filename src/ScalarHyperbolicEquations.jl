
"""
    ScalarHyperbolicEquation

Each scalar hyperbolic equation is a struct that is a subtype of the abstract type ScalarHyperbolicEquation.
Each struct then overrides the velocity function which evaluates the velocity f'(u, x, t). See LinearAdvection and 
BurgersEquation for examples.
"""
abstract type ScalarHyperbolicEquation <: HyperbolicPDE end
abstract type LinearScalarHyperbolicEquation <: ScalarHyperbolicEquation end
abstract type NonLinearScalarHyperbolicEquation <: ScalarHyperbolicEquation end

struct LinearAdvection{T} <: LinearScalarHyperbolicEquation
    vel::T

    function LinearAdvection(vel::Union{Real, Tuple{<:Real, <:Real}})
        if vel isa Real
            new{Float64}(convert(Float64, vel))
        else
            new{Tuple{Float64, Float64}}((convert(Float64, vel[1]), convert(Float64, vel[2])))
        end
    end
end

@inline function velocity(eq::LinearAdvection, rho::Real)
    return eq.vel
end

@inline function flux(eq::LinearAdvection{T}, u::Real) where {T <: Real}
    return eq.vel*u
end

@inline function flux(eq::LinearAdvection{T}, u::Real) where {T <: Tuple{<:Real, <:Real}}
    return (eq.vel[1]*u, eq.vel[2]*u)
end


struct BurgersEquation <: NonLinearScalarHyperbolicEquation end

@inline function velocity(eq::BurgersEquation, u::Real)
    return u
end

@inline function flux(eq::BurgersEquation, u::Real)
    return (u^2)/2
end

struct BurgersEquation2D <: NonLinearScalarHyperbolicEquation end

@inline function velocity(eq::BurgersEquation2D, u::Real)
    return (u,u)
end

@inline function flux(eq::BurgersEquation2D, u::Real)
    return ((u^2)/2,(u^2)/2)
end
