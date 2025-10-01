module Interpolations

using LinearAlgebra
using Statistics
using ..Meshfree4ScalarEq.ParticleGrids
using ..Meshfree4ScalarEq.SimSettings
using ..Meshfree4ScalarEq.HyperbolicPDEs
using ..Meshfree4ScalarEq.FluxFunctions

export functionInterpolation!, gradInterpolation!, setCurvatures!, GradientInterpolator, initTimeStep, UpwindGradient, CentralGradient, WENO, MUSCL, AxelMUSCL, DumbserWENO, MLSWeightFunction, inverseWeightFunction, exponentialWeightFunction, getStencil, LaxFriedrichsGradient, MUSCLlimited,
       AbstractSlopeLimiter, BarthJespersenLimiter, VenkatakrishnanLimiter, SuperbeeLimiter, MinmodLimiter, NoLimiter  

"""
    sortFlux(flux_ij::Real, flux_ji::Real, deltaX::Real)::Tuple{<:Real, <:Real}

Given a reconstruction of the state at the midpoint from the cell center flux1, and a state reconstruction from the neighbouring point, return the left and right state based on the relative orientation of the points.
"""
function sortFlux(flux_ij::Float64, flux_ji::Float64, deltaX::Float64)::Tuple{Float64, Float64}
    if deltaX > 0.0
        return (flux_ij, flux_ji)  # left state, right state
    else
        return (flux_ji, flux_ij)
    end
end

"""
    sortFlux(flux_ij::Real, flux_ji::Real, deltaX::Real)::Tuple{<:Real, <:Real}

Given a reconstruction of the state at the midpoint from the cell center flux1, and a state reconstruction from the neighbouring point, return the left and right state in x and y direction.
"""
function sortFlux(flux_ij::Float64, flux_ji::Float64, deltaX::Float64, deltaY::Float64)::Tuple{Float64, Float64, Float64, Float64}
    if deltaX > 0.0 && deltaY > 0.0
        return (flux_ij, flux_ji, flux_ij, flux_ji)
    elseif deltaX > 0.0 && deltaY < 0.0 
        return (flux_ij, flux_ji, flux_ji, flux_ij)
    elseif deltaX < 0.0 && deltaY > 0.0
        return (flux_ji, flux_ij, flux_ij, flux_ji)
    else
        return (flux_ji, flux_ij, flux_ji, flux_ij)
    end
end

# Weightfunction logic
abstract type MLSWeightFunction end
struct exponentialWeightFunction <: MLSWeightFunction end
struct inverseWeightFunction <: MLSWeightFunction end

@inline function (w::exponentialWeightFunction)(dxVec; param::Real, normalisation::Real)
    return @. exp(-param*((dxVec/normalisation)^2))
end

@inline function (w::exponentialWeightFunction)(dxVec, dyVec; param::Real, normalisation::Real)
    return @. exp(-param*((dxVec^2 + dyVec^2)/(normalisation^2)))
end

@inline function (w::inverseWeightFunction)(dxVec; param::Real, normalisation::Real)
    return @. 1/(dxVec^2)
end

@inline function (w::inverseWeightFunction)(dxVec, dyVec; param::Real, normalisation::Real)
    return @. 1/(dxVec^2 + dyVec^2)
end

"""
    functionInterpolation!(dxVec::AV1, wVec::AV2, fVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real}}
Polynomial interpolation of the function based on Taylor-Polynomial least squares method. The LS system is solved by solving the normal equations. In case of bad conditioning, scale dxVec or solve LS system using QR method.

wVec and res are overwritten.

# Arguments:
- `dxVec::AbstractVector`: Vector that contains x(j)-x(i) with x(i) the point at which we want to commpute the function value and x(j) the surrounding interpolation points.
- `wVec::AbstractVector`: Weight vector in Least squares problem. 
- `fVec::AbstractVector`: Function values at points x(j).
- `res::AbstractVector`: Vector that will contains the result. res[1] is always the interpolated value, res[2] the first order derivative at x(i), res[3] the second order derivative ... Derivatives are only returned if the order is high enough.
- `order::Int64`: Order of polynomial approximation.
"""
function functionInterpolation!(dxVec::AV1, wVec::AV2, fVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{Float64}, AV2 <: AbstractVector{Float64},
                                                                                                AV3 <: AbstractVector{Float64}, AV4 <: AbstractVector{Float64}}
    @assert length(dxVec) == length(wVec)
    @assert length(wVec) == length(fVec)
    @assert length(res) >= order + 1
    @assert length(wVec) >= order + 1 "At least $(order+1) points need for order $(order) LS interpolation. Only $(length(wVec)) points given."
    @assert !any(isnan, fVec)
    @assert !any(isnan, dxVec)

    if order == 0
        res[1] = dot(wVec, fVec) / sum(wVec)
    elseif order == 1
        b1 = dot(fVec, wVec)
        A11 = sum(wVec)
        wVec .= wVec .* dxVec  # wVec = dx .* wVec
        b2 = dot(fVec, wVec)
        A12 = sum(wVec)
        wVec .= wVec .* dxVec  # wVec = dx.^2 .* wVec
        A22 = sum(wVec)
        res[1] = (b2 - A22*b1/A12)/(A12 - A22*A11/A12)
        res[2] = (b1 - A11*res[1])/A12
    elseif order == 2
        # Generate normal equations
        b1 = dot(wVec, fVec)
        A11 = sum(wVec)
        wVec .= wVec .* dxVec  # wVec = dx .* wVec
        A12 = sum(wVec)
        b2 = dot(wVec, fVec)
        wVec .= wVec .* dxVec  # wVec = dx.^2 .* wVec
        A22 = sum(wVec)
        A13 = A22/2
        b3 = dot(wVec, fVec)/2
        wVec .= wVec .* dxVec  # wVec = dx.^3 .* wVec
        A23 = sum(wVec)/2
        A33 = dot(wVec, dxVec)/4

        # Hardcoded solve of 3x3 LU method
        L21 = A12/A11
        L31 = A13/A11
        U22 = A22-L21*A12
        L32 = (A23-L31*A12)/U22
        U23 = A23-L21*A13
        U33 = A33 - L31*A13 - L32*U23
        y2 = b2 - L21*b1
        y3 = b3 - L31*b1 - L32.*y2

        res[3] = y3 / U33
        res[2] = (y2 - U23*res[3]) / U22
        res[1] = (b1 - A12*res[2] - A13*res[3]) / A11
    else
        error("Order not implemented.")
    end
    @assert !any(isnan, res) "Function contains NaN's in functionInterpolation! method."
end

"""
    gradInterpolation!(dxVec::AV1, wVec::AV2, dfVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real}}
Same as `functionInterpolation!` but interpolation for the gradient. dfVec now contains (f(x(j))-f(x(i)), ...)

"""
function gradInterpolation!(dxVec::AV1, wVec::AV2, dfVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{Float64}, AV2 <: AbstractVector{Float64},
                                                                                                AV3 <: AbstractVector{Float64}, AV4 <: AbstractVector{Float64}}
    @assert length(dxVec) == length(wVec)
    @assert length(wVec) == length(dfVec)
    @assert length(res) >= order
    @assert length(wVec) >= order "At least $(order) points need for order $(order) LS interpolation. Only $(length(wVec)) points given."
    @assert !any(isnan, dfVec)
    @assert !any(isnan, dxVec)

    if order == 1
        wVec .= wVec .* dxVec  # wVec = dx .* wVec
        b1 = dot(dfVec, wVec)
        @assert !isnan(b1) "$(b1), $(dfVec), $(wVec)"
        A11 = dot(wVec, dxVec)
        res[1] = b1/A11
        @assert !any(isnan, res[1]) "Gradient contains NaN's in gradInterpolation! method. $(res), $(A11), $(dfVec), $(wVec), $(dxVec)"
        @assert !any(isinf, res[1]) "Gradient contains Inf's in gradInterpolation! method. $(res), $(A11), $(dfVec), $(wVec), $(dxVec)"
    elseif order == 2
        # Generate normal equations
        wVec .= wVec .* dxVec  # wVec = dx .* wVec
        b2 = dot(wVec, dfVec)
        wVec .= wVec .* dxVec  # wVec = dx.^2 .* wVec
        A11 = sum(wVec)
        b3 = dot(wVec, dfVec)/2
        wVec .= wVec .* dxVec  # wVec = dx.^3 .* wVec
        A12 = sum(wVec)/2
        A22 = dot(wVec, dxVec)/4

        # Explicit solve of 2x2 linear system
        res[1] = (b3*A12 - A22*b2)/((A12^2) - A22*A11)
        res[2] = (b3 - A12*res[1])/A22
        @assert !any(isnan, res[1:2]) "Gradient contains NaN's in gradInterpolation! method. $(res), $((A12^2) - A22*A11), $(dfVec)"
        @assert !any(isinf, res[1:2]) "Gradient contains Inf's in gradInterpolation! method. $(res), $((A12^2) - A22*A11), $(dfVec)"
    else
        error("Order not implemented.")
    end
end

"""
    gradInterpolation!(dxVec::AV1, wVec::AV2, dfVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real}}
2D Gradient interpolation. At least 2 points required for order 1 (res should of length 2). At least 5 points required for order 2 (res should be of length five).
"""
function gradInterpolation!(dxVec::AV1, dyVec::AV5, wVec::AV2, dfVec::AV3, res::AV4; order::Int64=1) where {    AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real},
                                                                                                                AV5 <: AbstractVector{<:Real}}
    @assert length(dxVec) == length(wVec) == length(dyVec)
    @assert length(wVec) == length(dfVec)
    @assert !any(isnan, dfVec)
    @assert !any(isnan, dxVec)
    @assert !any(isnan, dyVec)

    if order == 1
        @assert length(res) == 2
        # Create 2x2 linear system
        A11 = A12 = A22 = b1 = b2 = 0.0
        for (w, dx) in zip(wVec, dxVec)
            A11 += w*(dx^2)
        end
        for (w, dy) in zip(wVec, dyVec)
            A22 += w*(dy^2)
        end
        for (w, dx, dy) in zip(wVec, dxVec, dyVec)
            A12 += w*dx*dy
        end
        for (w, dx, df) in zip(wVec, dxVec, dfVec)
            b1 += w*dx*df
        end
        for (w, dy, df) in zip(wVec, dyVec, dfVec)
            b2 += w*dy*df
        end
        # Explicit solve of 2x2 linear system
        res[1] = (b2*A12 - A22*b1)/((A12^2) - A22*A11)
        res[2] = (b2 - A12*res[1])/A22
        @assert !any(isnan, res) "Gradient contains NaN's in gradInterpolation! method. $(res), $((A12^2) - A22*A11), $(dfVec), $(dxVec), $(dyVec)"
    elseif order == 2
        @assert length(res) == 5
        # Solve LS problem using the robust pseudo-inverse `pinv`.
        A = similar(dxVec, length(dxVec), 5) # Use a local temporary matrix
        @. A[:, 1] = dxVec * wVec
        @. A[:, 2] = dyVec * wVec
        @. A[:, 3] = (dxVec^2) * wVec / 2
        @. A[:, 4] = (dyVec^2) * wVec / 2
        @. A[:, 5] = dxVec * dyVec * wVec
        
        # Use pinv for robustness against singular stencils
        res .= pinv(A, rtol = sqrt(eps(real(float(oneunit(eltype(A))))))) * (wVec .* dfVec)
    else
        error("Order not implemented.")
    end
end

"""
    GradientInterpolator

In case of unstructured grids, the spatial gradient is approximated using a moving least squares (MLS) method based on Taylor polynomials.
These algorithms are implemented as follows. Each method is a struct that is a subtype of GradientInterpolator. The gradient 
at a gridpoint can then be computed using the ()-operator; see for example UpwindGradient and CentralGradient. These objects select
the correct stencil and then call the MLS routine (gradInterpolation).
"""
abstract type GradientInterpolator end

function initTimeStep(g::GradientInterpolator, particleGrid::ParticleGrid, interpAlpha::Real, interpRange::Real) end  # Function called at the start of a time step (order RK-stage)

# # ------------------------------- Dumbser WENO -------------------------------

# function getStencil(deltaX::Real, deltaY::Real, s::Int64)
#     stencil = convert(Int64, div(s*(atan(deltaY, deltaX) + pi)*4/pi, s))
#     stencil = stencil == 8 ? 0 : stencil  # Negative x-axis should be contained in stencil 0
#     return stencil
# end

# struct DumbserWENO <: GradientInterpolator
#     order::Int64
#     res::Vector{Float64}
#     weightFunction::MLSWeightFunction
#     s::Integer  # amount of one-sided stencils
#     gradients::Matrix{Float64}
#     weights::Vector{Float64}

#     function DumbserWENO(order::Int64 = 2; weightFunction::MLSWeightFunction = exponentialWeightFunction())
#         @assert order == 2 "Order must be to two, since the WENO weights require a second derivative."
#         new(order, Vector{Float64}(undef, 5), weightFunction, 8, Matrix{Float64}(undef, (5, 9)), Vector{Float64}(undef, 9))
#     end
# end

# function (weno::DumbserWENO)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, eq::LinearAdvection{2}, settings::SimSetting; setCurvature::Bool=true)::Real
#     @assert settings.interpRange >= sqrt(5.0^2 + 3.0^2)*particleGrid.dx "Interpolation must be sufficiently larger, otherwise one cannot guarantee sufficient neighbours are found." 
#     particle = particleGrid.grid[particleIndex]
#     Npts = length(particle.neighbourIndices)

#     # Divide points in stencils
#     windowMatrix = zeros(Bool, (Npts, weno.s+1))
#     windowMatrix[:, 1] .= true  # First column is the central stencil

#     for i in eachindex(particle.neighbourIndices)
#         nbIndex = particle.neighbourIndices[i]
#         deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
#         particle.dxVec[i] = deltaX/settings.interpRange
#         particle.dyVec[i] = deltaY/settings.interpRange
#         particle.dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
#         stencil = getStencil(deltaX, deltaY, weno.s)  # in [0, 7]
#         windowMatrix[i, stencil+2] = true
#     end
#     for stencil in 1:weno.s+1
#         particle.wVec .= weno.weightFunction(particle.dxVec, particle.dyVec; param=settings.interpAlpha, normalisation=1.0)
        
#         # There should be at least 5 points in each stencil!
#         @assert count(windowMatrix[:, stencil]) >= 5 "($(particle.pos[1]), $(particle.pos[2])), $(count(windowMatrix[:, stencil])), $(stencil)"
#         gradInterpolation!(particle.dxVec[windowMatrix[:, stencil]], particle.dyVec[windowMatrix[:, stencil]], particle.wVec[windowMatrix[:, stencil]], particle.dfVec[windowMatrix[:, stencil]], weno.res; order=weno.order)

#         # Rescale results
#         weno.gradients[1, stencil] = weno.res[1]/settings.interpRange
#         weno.gradients[2, stencil] = weno.res[2]/settings.interpRange  
#         weno.gradients[3, stencil] = weno.res[3]/(settings.interpRange^2)
#         weno.gradients[4, stencil] = weno.res[4]/(settings.interpRange^2)
#         weno.gradients[5, stencil] = weno.res[5]/(settings.interpRange^2)

#         # Compute weights
#         r = 4
#         eps = 1e-14
#         lambda = (stencil == 1) ? 10^5 : 1.0
#         weno.weights[stencil] = lambda/((eps + sum((x^2 for x in weno.gradients[:, stencil])))^r)
#     end

#     # Normalise weights
#     weno.weights .= weno.weights ./ sum(weno.weights)
    
#     if setCurvature 
#         particle.curvature[1] = 0.0
#         particle.curvature[2] = 0.0
#         for i in eachindex(weno.weights)  # Write out inner product
#             particle.curvature[1] += weno.weights[i]*weno.gradients[3, i]
#             particle.curvature[2] += weno.weights[i]*weno.gradients[4, i]
#         end
#     end

#     # Compute divergence
#     res = 0.0
    
#     for i in eachindex(weno.weights)  # Write out inner product
#         res += weno.weights[i]*(weno.gradients[1, i]*eq.vel[1] + eq.vel[2]*weno.gradients[2, i])
#     end
#     return res
# end

include("./CentralGradient.jl")
include("./MUSCL.jl")
include("./Upwind.jl")
include("./WENO.jl")

end # End Module Interpolations