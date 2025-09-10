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
function sortFlux(flux_ij::Real, flux_ji::Real, deltaX::Real)::Tuple{<:Real, <:Real}
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
        # Solve LS problem using Julia's backslash operator. Requires an allocation (A), but condition number doesn't square!
        A = Matrix{Float64}(undef, length(dxVec), 5)
        @. A[:, 1] = dxVec * wVec
        @. A[:, 2] = dyVec * wVec
        @. A[:, 3] = (dxVec^2) * wVec / 2
        @. A[:, 4] = (dyVec^2) * wVec / 2
        @. A[:, 5] = dxVec * dyVec * wVec
        wVec .= wVec .* dfVec
        res .= A \ wVec
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

# ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  # Only relevant in 2D. In 1D, all algorithms are the same.
abstract type TiwariAlgorithm <: UpwindAlgorithm end  # Split domain in left and right for d/dx, and up and down for d/dy.
abstract type PraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type ClassicAlgorithm <: UpwindAlgorithm end  # Take all points 'behind' center point. 
abstract type RusanovAlgorithm <: UpwindAlgorithm end # This is no upwinding of course but easy implementation in this framework (numerical Flux given does not have to be upwind)
struct UpwindGradient{Algorithm} <: GradientInterpolator where {Algorithm <: UpwindAlgorithm}
    order::Int64
    res::Vector{Float64}
    weightFunction::MLSWeightFunction
    numericalFlux::NumericalFluxFunction

    """
        UpwindGradient(order::Int64 = 1; algType::String = "")

    Constructor for Upwind Object. algType only has impact in 2D upwinding.
    """
    function UpwindGradient(order::Int64 = 1; numericalFlux::NumericalFluxFunction = UpwindFlux(), algType::String = "Classic", weightFunction::MLSWeightFunction = exponentialWeightFunction())
        @assert order >= 1 "Order must be larger or equal to one."
        @assert algType in ["Classic", "Tiwari", "Praveen", "NonLinearPraveen"]
        if order == 1
            size = 2  # In 2D res has length 2, in 1D res has length 1
        elseif order == 2
            size = 5  # In 2D res had length 5, in 1D res has length 2
        end
        if algType == "Classic"
            new{ClassicAlgorithm}(order, Vector{Float64}(undef, size), weightFunction, numericalFlux)
        elseif algType == "Praveen"
            @assert order == 1
            new{PraveenAlgorithm}(order, Vector{Float64}(undef, size), weightFunction, numericalFlux)
        elseif algType == "NonLinearPraveen"
            @assert order == 1
            new{NonLinearPraveenAlgorithm}(order, Vector{Float64}(undef, size), weightFunction, numericalFlux)
        elseif algType == "Tiwari"
            new{TiwariAlgorithm}(order, Vector{Float64}(undef, size), weightFunction, numericalFlux)
        end
    end
end

function (upwind::UpwindGradient)(particleGrid::ParticleGrid1D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::ScalarHyperbolicEquation{D}, settings::SimSetting; setCurvature::Bool=true)::Real where {D}
    nbNeighbours = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, nbNeighbours)
    dfVec = Vector{Float64}(undef, nbNeighbours)
    for (index, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
        deltaPos = getDistance(particleGrid, particleIndex, nbIndex)
        fm, fp = sortFlux(fVec[particleIndex], fVec[nbIndex], deltaPos)
        dxVec[index] = deltaPos/settings.interpRange
        dfVec[index] = upwind.numericalFlux(fm, fp, eq) - flux(eq, fVec[particleIndex])
    end
    wVec = upwind.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)
    @assert !any(isnan, wVec) && !any(isinf, wVec) "Infs or Nan's in wVec: $(wVec)"

    gradInterpolation!(dxVec, wVec, dfVec, upwind.res; order=upwind.order)

    if setCurvature
        particleGrid.grid[particleIndex].curvature = 0.0
    end
    return 2*upwind.res[1]/settings.interpRange
end

# deprecated
# struct LaxFriedrichsGradient <: GradientInterpolator
#     res::Vector{Float64}
#     weightFunction::MLSWeightFunction
#     numericalFlux::NumericalFluxFunction

#     function LaxFriedrichsGradient(; weightFunction::MLSWeightFunction = exponentialWeightFunction())
#         new(Vector{Float64}(undef, 2), weightFunction, LaxFriedrichsFlux())
#     end
# end

# function (laxFriedrichs::LaxFriedrichsGradient)(particleGrid::ParticleGrid1D, particleIndex::Integer, fVec::Vector{<:Real}, eq::ScalarHyperbolicEquation, settings::SimSetting; setCurvature::Bool=true)::Real
#     nbNeighbours = length(particleGrid.grid[particleIndex].neighbourIndices)
#     dxVec = Vector{Float64}(undef, nbNeighbours)
#     dfVec = Vector{Float64}(undef, nbNeighbours)
#     maxFlux = maximum(map(particle -> velocity(eq, particle.rho), particleGrid.grid))
#     for (index, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
#         deltaPos = getDistance(particleGrid, particleIndex, nbIndex)
#         fm, fp = sortFlux(fVec[particleIndex], fVec[nbIndex], deltaPos)
#         dxVec[index] = deltaPos/settings.interpRange
#         dfVec[index] = laxFriedrichs.numericalFlux(fm, fp, eq, maxFlux) - flux(eq, fVec[particleIndex])
#     end
#     wVec = laxFriedrichs.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)
#     @assert !any(isnan, wVec) && !any(isinf, wVec) "Infs or Nan's in wVec: $(wVec)"

#     gradInterpolation!(dxVec, wVec, dfVec, laxFriedrichs.res; order=1)

#     if setCurvature
#         particleGrid.grid[particleIndex].curvature = 0.0
#     end
#     return 2*laxFriedrichs.res[1]/settings.interpRange
# end

function (upwind::UpwindGradient{TiwariAlgorithm})(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::LinearAdvection{2}, settings::SimSetting; setCurvature::Bool=true)::Real    
    vel = eq.vel
    nbNeighbours = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, nbNeighbours)
    dyVec = Vector{Float64}(undef, nbNeighbours)
    dfVec = Vector{Float64}(undef, nbNeighbours)    
    xWindow = Vector{Bool}(undef, nbNeighbours)  # True if point should be used for d/dx
    yWindow = Vector{Bool}(undef, nbNeighbours)  # True if point should be used for d/dx

    for (i, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX/settings.interpRange
        dyVec[i] = deltaY/settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        xWindow[i] = ((vel[1] >= 0.0) && (deltaX <= 0.0)) || ((vel[1] <= 0.0) && (deltaX >= 0.0))
        yWindow[i] = ((vel[2] >= 0.0) && (deltaY <= 0.0)) || ((vel[2] <= 0.0) && (deltaY >= 0.0))
    end
    wVec = upwind.weightFunction(dxVec[xWindow], dyVec[xWindow]; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec[xWindow], dyVec[xWindow], wVec, dfVec[xWindow], upwind.res; order=upwind.order)
    ddx = upwind.res[1]/settings.interpRange

    if upwind.order == 1  # Set the curvature
        particleGrid.grid[particleIndex].curvature[1] = 0.0
    elseif upwind.order == 2
        particleGrid.grid[particleIndex].curvature[1] = upwind.res[3]/(settings.interpRange^2)
    end
        
    wVec = upwind.weightFunction(dxVec[yWindow], dyVec[yWindow]; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec[yWindow], dyVec[yWindow], wVec, dfVec[yWindow], upwind.res; order=upwind.order)
    ddy = upwind.res[2]/settings.interpRange

    if setCurvature && (upwind.order == 1)
        particleGrid.grid[particleIndex].curvature[2] = 0.0
    elseif setCurvature && (upwind.order == 2)
        particleGrid.grid[particleIndex].curvature[2] = upwind.res[4]/(settings.interpRange^2)
    end

    return ddx*vel[1] + ddy*vel[2]
end

function (upwind::UpwindGradient{ClassicAlgorithm})(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::LinearAdvection{2}, settings::SimSetting; setCurvature::Bool=true)::Real    
    vel = eq.vel
    dxVec = Vector{Float64}(undef, 0)
    dyVec = Vector{Float64}(undef, 0)
    dfVec = Vector{Float64}(undef, 0)    
    for nbIndex in particleGrid.grid[particleIndex].neighbourIndices
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        if deltaX*vel[1] + deltaY*vel[2] < 0
            push!(dxVec, deltaX/settings.interpRange)
            push!(dyVec, deltaY/settings.interpRange)
            push!(dfVec, fVec[nbIndex] - fVec[particleIndex])
        end
    end
    wVec = upwind.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec, dyVec, wVec, dfVec, upwind.res; order=upwind.order)

    if setCurvature && (upwind.order == 1) 
        particleGrid.grid[particleIndex].curvature[1] = 0.0
        particleGrid.grid[particleIndex].curvature[2] = 0.0
    elseif setCurvature && (upwind.order == 2)
        particleGrid.grid[particleIndex].curvature[1] = upwind.res[3]/(settings.interpRange^2)
        particleGrid.grid[particleIndex].curvature[2] = upwind.res[4]/(settings.interpRange^2)
    end
    
    return vel[1]*upwind.res[1]/settings.interpRange + vel[2]*upwind.res[2]/settings.interpRange
end

function (upwind::UpwindGradient{PraveenAlgorithm})(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::LinearAdvection{2}, settings::SimSetting; setCurvature::Bool=true)::Real    
    particle = particleGrid.grid[particleIndex]
    vel = eq.vel
    if setCurvature
        particle.curvature[1] = 0.0
        particle.curvature[2] = 0.0
    end
    
    # Create 2x2 LS system
    A11 = A12 = A22 = 0.0
    for nbIndex in particle.neighbourIndices
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)
        A11 += w*(deltaX^2) 
        A12 += w*deltaX*deltaY
        A22 += w*(deltaY^2)
    end
    D = A11*A22 - (A12^2)
    
    div = 0.0
    for nbIndex in particle.neighbourIndices

        # Solve 2x2 LS system
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)
        coeff = ((A22*w*deltaX - A12*w*deltaY)/D, (A11*w*deltaY - A12*w*deltaX)/D)

        # Compute adapted coefficients
        angle = atan(deltaY, deltaX)  # atan2 function
        n = (cos(angle), sin(angle))
        s = (-sin(angle), cos(angle))
        alfaBar = dot(n, coeff)
        betaBar = dot(s, coeff)
        bracketMinus = dot(vel, n) > 0.0 ? 0.0 : dot(vel, n)
        bracketMinus2 = betaBar*dot(vel, s) > 0.0 ? 0.0 : betaBar*dot(vel, s)
        cij = alfaBar*bracketMinus + bracketMinus2
        div += 2*cij*(fVec[nbIndex] - fVec[particleIndex])
    end
    return div
end

function (upwind::UpwindGradient{NonLinearPraveenAlgorithm})(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, eq::ScalarHyperbolicEquation{D}, settings::SimSetting; setCurvature::Bool=true)::Real where {D}   
    particle = particleGrid.grid[particleIndex]
    vel = eq.vel
    if setCurvature
        particle.curvature[1] = 0.0
        particle.curvature[2] = 0.0
    end
    
    # Create 2x2 LS system
    A11 = A12 = A22 = 0.0
    for nbIndex in particle.neighbourIndices
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)
        A11 += w*(deltaX^2) 
        A12 += w*deltaX*deltaY
        A22 += w*(deltaY^2)
    end
    D = A11*A22 - (A12^2)
    
    
    ui = fVec[particleIndex]
    fiVec = flux(eq, ui)
    
    div = 0.0
    for nbIndex in particle.neighbourIndices
        uj = fVec[nbIndex]

        # Solve 2x2 LS system
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)
        coeff = ((A22*w*deltaX - A12*w*deltaY)/D, (A11*w*deltaY - A12*w*deltaX)/D)

        # Compute adapted coefficients
        angle = atan(deltaY, deltaX)  # atan2 function
        n = (cos(angle), sin(angle))
        s = (-sin(angle), cos(angle))

        # Numerical flux at point i
        Fni = dot(fiVec, n)  # Numerical flux along normal at point i
        Gsi = dot(fiVec, s)  # Numerical flux along s at point i

        # Compute numerical flux at midpoint
        fjVec = flux(eq, uj)
        aij = ((fjVec[1] - fiVec[1])/(uj- ui), (fjVec[2] - fiVec[2])/(uj- ui))
        favg = ((fjVec[1] + fiVec[1])/2, (fjVec[2] + fiVec[2])/2)
        Fnij = dot(favg, n) - abs(dot(aij, n))*(uj - ui)/2  # Numerical flux along normal at midpoint
        Gsij = dot(favg, s) - sign(dot(s, coeff))*abs(dot(aij, s))*(uj - ui)/2  # Numerical flux along s at midpoint

        Fndiff = Fnij - Fni
        Gsdiff = Gsij - Gsi

        # Compute flux difference in x and y direction (rotation back)
        Fdiff = n[1]*Fndiff + s[1]*Gsdiff
        Gdiff = n[2]*Fndiff + s[2]*Gsdiff

        div += coeff[1]*Fdiff + coeff[2]*Gdiff
    end
    return 2*div
end

# ------------------------------- CentralGradient -------------------------------
struct CentralGradient <: GradientInterpolator
    order::Int64
    res::Vector{Float64}
    weightFunction::MLSWeightFunction

    function CentralGradient(order::Int64 = 1; weightFunction::MLSWeightFunction = exponentialWeightFunction())
        @assert order >= 1 "Order must be larger or equal to one."
        if order == 1
            size = 2  # In 2D res has length 2, in 1D res has length 1
        elseif order == 2
            size = 5  # In 2D res had length 5, in 1D res has length 2
        end
        new(order, Vector{Float64}(undef, size), weightFunction)
    end
end

function (central::CentralGradient)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::AbstractArray{<:Real}, eq::LinearAdvection{2}, settings::SimSetting; setCurvature::Bool=true)::Real
    Npts = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, Npts)
    dyVec = Vector{Float64}(undef, Npts)
    dfVec = Vector{Float64}(undef, Npts)

    for i in eachindex(particleGrid.grid[particleIndex].neighbourIndices)
        nbIndex = particleGrid.grid[particleIndex].neighbourIndices[i]
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX/particleGrid.dx
        dyVec[i] = deltaY/particleGrid.dx
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end
    wVec = central.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)

    gradInterpolation!(dxVec, dyVec, wVec, dfVec, central.res; order=central.order)

    if setCurvature && (central.order == 1)
        particleGrid.grid[particleIndex].curvature[1] = 0.0
        particleGrid.grid[particleIndex].curvature[2] = 0.0
    elseif setCurvature && (central.order == 2)
        particleGrid.grid[particleIndex].curvature[1] = central.res[3]/(particleGrid.dx^2)
        particleGrid.grid[particleIndex].curvature[2] = central.res[4]/(particleGrid.dx^2)
    end

    return eq.vel[1]*central.res[1]/particleGrid.dx + eq.vel[2]*central.res[2]/particleGrid.dx
end

function (central::CentralGradient)(particleGrid::ParticleGrid1D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::LinearAdvection{1}, settings::SimSetting; setCurvature::Bool=true)::Real
    Npts = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, Npts)
    dfVec = Vector{Float64}(undef, Npts)

    for i in eachindex(particleGrid.grid[particleIndex].neighbourIndices)
        nbIndex = particleGrid.grid[particleIndex].neighbourIndices[i]
        deltaPos = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaPos/particleGrid.dx
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end
    wVec = central.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)

    gradInterpolation!(dxVec, wVec, dfVec, central.res; order=central.order)

    if setCurvature && (central.order == 1)
        particleGrid.grid[particleIndex].curvature = 0.0
    elseif setCurvature && (central.order == 2)
        particleGrid.grid[particleIndex].curvature = central.res[2]/(particleGrid.dx^2)
    end

    return eq.vel*central.res[1]/particleGrid.dx
end

# ------------------------------- WENO -------------------------------
struct WENO <: GradientInterpolator
    order::Int64
    res::Vector{Float64}
    weightFunction::MLSWeightFunction

    function WENO(order::Int64 = 1; weightFunction::MLSWeightFunction = exponentialWeightFunction())
        @assert order >= 2 "Order must be larger or equal to two, since the WENO weights require a second derivative."
        if order == 1
            size = 2  # In 2D res has length 2, in 1D res has length 1
        elseif order == 2
            size = 5  # In 2D res had length 5, in 1D res has length 2
        end
        new(order, Vector{Float64}(undef, size), weightFunction)
    end
end

function (weno::WENO)(particleGrid::ParticleGrid1D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::LinearAdvection{1}, settings::SimSetting; setCurvature::Bool=true)::Real
    Npts = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, Npts)
    dfVec = Vector{Float64}(undef, Npts)
    leftWindow = Vector{Bool}(undef, Npts)
    for i in eachindex(particleGrid.grid[particleIndex].neighbourIndices)
        nbIndex = particleGrid.grid[particleIndex].neighbourIndices[i]
        deltaPos = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaPos
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        leftWindow[i] = deltaPos > 0.0 ? false : true
    end
    wVec = weno.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)

    # One-sided stencil
    if eq.vel > 0.0
        # Left stencil
        gradInterpolation!(dxVec[leftWindow], wVec[leftWindow], dfVec[leftWindow], weno.res; order=weno.order)
    else
        # Right stencil
        gradInterpolation!(dxVec[.!leftWindow], wVec[.!leftWindow], dfVec[.!leftWindow], weno.res; order=weno.order)
    end
    resS1 = weno.res[1]
    resS2 = weno.res[2]

    # Central stencil
    wVec .= weno.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec, wVec, dfVec, weno.res; order=weno.order)
    resC1 = weno.res[1]
    resC2 = weno.res[2]

    e = 1e-6
    dx2 = particleGrid.dx^2
    dx4 = dx2^2
    betaS = 0.5/(((resS1^2)*dx2 + (resS2^2)*dx4 + e)^2)
    betaC = 0.5/(((resC1^2)*dx2 + (resC2^2)*dx4 + e)^2)
    ω_s = betaS/(betaC + betaS)
    ω_c = betaC/(betaC + betaS)

    if setCurvature
        particleGrid.grid[particleIndex].curvature = resS2*ω_s + resC2*ω_c
    end

    res = resS1*ω_s + resC1*ω_c
    @assert !isnan(res) "$(weno.res), $(weno.res), $(betaS), $(betaC), $(ω_s), $(ω_c), $(dfVec)"
    return res*eq.vel
end

function (weno::WENO)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, eq::LinearAdvection{2}, settings::SimSetting; setCurvature::Bool=true)::Real
    Npts = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, Npts)
    dyVec = Vector{Float64}(undef, Npts)
    dfVec = Vector{Float64}(undef, Npts)
    leftWindow = Vector{Bool}(undef, Npts)
    topWindow = Vector{Bool}(undef, Npts)
    for i in eachindex(particleGrid.grid[particleIndex].neighbourIndices)
        nbIndex = particleGrid.grid[particleIndex].neighbourIndices[i]
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX/settings.interpRange
        dyVec[i] = deltaY/settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        leftWindow[i] = deltaX > 0.0 ? false : true
        topWindow[i] = deltaY > 0.0 ? true : false
    end
    wVec = weno.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)

    # One-sided stencil - Left & Right
    if eq.vel[1] > 0.0
        # Left stencil
        gradInterpolation!(dxVec[leftWindow], dyVec[leftWindow], wVec[leftWindow], dfVec[leftWindow], weno.res; order=weno.order)
    else
        # Right stencil
        gradInterpolation!(dxVec[.!leftWindow], dyVec[.!leftWindow], wVec[.!leftWindow], dfVec[.!leftWindow], weno.res; order=weno.order)
    end
    resHx = weno.res[1]/settings.interpRange
    resHy = weno.res[2]/settings.interpRange
    resHxx = weno.res[3]/(settings.interpRange^2)
    resHyy = weno.res[4]/(settings.interpRange^2)
    resHxy = weno.res[5]/(settings.interpRange^2)
    
    # One-sided stencil - Up & Down
    wVec .= weno.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
    if eq.vel[2] < 0.0
        # Top stencil
        gradInterpolation!(dxVec[topWindow], dyVec[topWindow], wVec[topWindow], dfVec[topWindow], weno.res; order=weno.order)
    else
        # Bottom stencil
        gradInterpolation!(dxVec[.!topWindow], dyVec[.!topWindow], wVec[.!topWindow], dfVec[.!topWindow], weno.res; order=weno.order)
    end
    resVx = weno.res[1]/settings.interpRange
    resVy = weno.res[2]/settings.interpRange
    resVxx = weno.res[3]/(settings.interpRange^2)
    resVyy = weno.res[4]/(settings.interpRange^2)
    resVxy = weno.res[5]/(settings.interpRange^2)

    # Central stencil
    wVec .= weno.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec, dyVec, wVec, dfVec, weno.res; order=weno.order)
    resCx = weno.res[1]/settings.interpRange
    resCy = weno.res[2]/settings.interpRange
    resCxx = weno.res[3]/(settings.interpRange^2)
    resCyy = weno.res[4]/(settings.interpRange^2)
    resCxy = weno.res[5]/(settings.interpRange^2)

    # Compute non-linear weights
    e = 1e-12
    dx2 = particleGrid.dx^2
    dx4 = dx2^2
    betaH = 0.5/((resHx^2)*dx2 + (resHy^2)*dx2 + (resHxx^2)*dx4 + (resHyy^2)*dx4 + (resHxy^2)*dx4 + e)^2
    betaV = 0.5/((resVx^2)*dx2 + (resVy^2)*dx2 + (resVxx^2)*dx4 + (resVyy^2)*dx4 + (resVxy^2)*dx4 + e)^2
    betaC = 0.5/((resCx^2)*dx2 + (resCy^2)*dx2 + (resCxx^2)*dx4 + (resCyy^2)*dx4 + (resCxy^2)*dx4 + e)^2
    wH = betaH/(betaH + betaC)
    wCx = betaC/(betaH + betaC)
    wV = betaV/(betaC + betaV)
    wCy = betaC/(betaC + betaV)

    if setCurvature
        particleGrid.grid[particleIndex].curvature[1] = wH*resHxx + wCx*resCxx
        particleGrid.grid[particleIndex].curvature[2] = wV*resVyy + wCy*resCyy
    end

    return (wH*resHx + wCx*resCx)*eq.vel[1] + (wV*resVy + wCy*resCy)*eq.vel[2]
end


# ------------------------------- Dumbser WENO -------------------------------

function getStencil(deltaX::Real, deltaY::Real, s::Int64)
    stencil = convert(Int64, div(s*(atan(deltaY, deltaX) + pi)*4/pi, s))
    stencil = stencil == 8 ? 0 : stencil  # Negative x-axis should be contained in stencil 0
    return stencil
end

struct DumbserWENO <: GradientInterpolator
    order::Int64
    res::Vector{Float64}
    weightFunction::MLSWeightFunction
    s::Integer  # amount of one-sided stencils
    gradients::Matrix{Float64}
    weights::Vector{Float64}

    function DumbserWENO(order::Int64 = 2; weightFunction::MLSWeightFunction = exponentialWeightFunction())
        @assert order == 2 "Order must be to two, since the WENO weights require a second derivative."
        new(order, Vector{Float64}(undef, 5), weightFunction, 8, Matrix{Float64}(undef, (5, 9)), Vector{Float64}(undef, 9))
    end
end

function (weno::DumbserWENO)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, eq::LinearAdvection{2}, settings::SimSetting; setCurvature::Bool=true)::Real
    @assert settings.interpRange >= sqrt(5.0^2 + 3.0^2)*particleGrid.dx "Interpolation must be sufficiently larger, otherwise one cannot guarantee sufficient neighbours are found." 
    particle = particleGrid.grid[particleIndex]
    Npts = length(particle.neighbourIndices)

    # Divide points in stencils
    windowMatrix = zeros(Bool, (Npts, weno.s+1))
    windowMatrix[:, 1] .= true  # First column is the central stencil

    for i in eachindex(particle.neighbourIndices)
        nbIndex = particle.neighbourIndices[i]
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        particle.dxVec[i] = deltaX/settings.interpRange
        particle.dyVec[i] = deltaY/settings.interpRange
        particle.dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        stencil = getStencil(deltaX, deltaY, weno.s)  # in [0, 7]
        windowMatrix[i, stencil+2] = true
    end
    for stencil in 1:weno.s+1
        particle.wVec .= weno.weightFunction(particle.dxVec, particle.dyVec; param=settings.interpAlpha, normalisation=1.0)
        
        # There should be at least 5 points in each stencil!
        @assert count(windowMatrix[:, stencil]) >= 5 "($(particle.pos[1]), $(particle.pos[2])), $(count(windowMatrix[:, stencil])), $(stencil)"
        gradInterpolation!(particle.dxVec[windowMatrix[:, stencil]], particle.dyVec[windowMatrix[:, stencil]], particle.wVec[windowMatrix[:, stencil]], particle.dfVec[windowMatrix[:, stencil]], weno.res; order=weno.order)

        # Rescale results
        weno.gradients[1, stencil] = weno.res[1]/settings.interpRange
        weno.gradients[2, stencil] = weno.res[2]/settings.interpRange  
        weno.gradients[3, stencil] = weno.res[3]/(settings.interpRange^2)
        weno.gradients[4, stencil] = weno.res[4]/(settings.interpRange^2)
        weno.gradients[5, stencil] = weno.res[5]/(settings.interpRange^2)

        # Compute weights
        r = 4
        eps = 1e-14
        lambda = (stencil == 1) ? 10^5 : 1.0
        weno.weights[stencil] = lambda/((eps + sum((x^2 for x in weno.gradients[:, stencil])))^r)
    end

    # Normalise weights
    weno.weights .= weno.weights ./ sum(weno.weights)
    
    if setCurvature 
        particle.curvature[1] = 0.0
        particle.curvature[2] = 0.0
        for i in eachindex(weno.weights)  # Write out inner product
            particle.curvature[1] += weno.weights[i]*weno.gradients[3, i]
            particle.curvature[2] += weno.weights[i]*weno.gradients[4, i]
        end
    end

    # Compute divergence
    res = 0.0
    
    for i in eachindex(weno.weights)  # Write out inner product
        res += weno.weights[i]*(weno.gradients[1, i]*eq.vel[1] + eq.vel[2]*weno.gradients[2, i])
    end
    return res
end


# ------------------------------- MUSCL -------------------------------

abstract type MUSCLORDER end
struct MUSCLORDER1 <: MUSCLORDER end
struct MUSCLORDER2 <: MUSCLORDER end
struct MUSCLORDER3 <: MUSCLORDER end
struct MUSCLORDER4 <: MUSCLORDER end
"""
    MUSCL <: GradientInterpolator

Central reconstruction of any order with upwinding.
"""
mutable struct MUSCL{ORDER<:MUSCLORDER} <: GradientInterpolator
    order::ORDER
    res::Vector{Float64}  # Result of gradient computation
    weightFunction::MLSWeightFunction
    numericalFlux::NumericalFluxFunction

    function MUSCL(order::Int64; weightFunction::MLSWeightFunction = exponentialWeightFunction(), numericalFlux::NumericalFluxFunction = RusanovFlux())
        @assert (order == 1) || (order == 2) || (order == 3) || (order == 4) "Order must be one, two, three or four."
        if order == 1
            new{MUSCLORDER1}(MUSCLORDER1(), Vector{Float64}(undef, order), weightFunction, numericalFlux)
        elseif order == 2
            new{MUSCLORDER2}(MUSCLORDER2(), Vector{Float64}(undef, order), weightFunction, numericalFlux)
        elseif order == 3
            new{MUSCLORDER3}(MUSCLORDER3(), Vector{Float64}(undef, order), weightFunction, numericalFlux)
        elseif order == 4
            new{MUSCLORDER4}(MUSCLORDER4(), Vector{Float64}(undef, order), weightFunction, numericalFlux)
        end
    end
end

function initTimeStep(muscl::MUSCL{ORDER}, particleGrid::ParticleGrid1D, interpAlpha::Real, interpRange::Real) where {ORDER <: MUSCLORDER}
    # Compute reconstruction for particle in each neighbourhood and store gradient coefficients
    for (particleIndex, particle) in enumerate(particleGrid.grid)

        for (i, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
            deltaPos = getDistance(particleGrid, particleIndex, nbIndex)
            particle.dxVec[i] = deltaPos
        end
        particle.wVec .= muscl.weightFunction(particle.dxVec; param=interpAlpha, normalisation=particleGrid.dx)

        if ORDER == MUSCLORDER1
            @. particle.wVec = particle.wVec * particle.dxVec  # wVec .= dx .* wVec
            t = dot(particle.wVec, particle.dxVec)
            @. particle.alfaij = particle.wVec / t  # Derivative
        elseif ORDER == MUSCLORDER2
            # alfa_ij
            @. particle.wVec = particle.wVec * particle.dxVec  # wVec .= dx .* wVec
            t = dot(particle.wVec, particle.dxVec)
            @. particle.alfaij = particle.wVec / t

            # Restore vector
            particle.wVec .= muscl.weightFunction(particle.dxVec; param=interpAlpha, normalisation=particleGrid.dx)

            # alfa_ijBar and betaij
            @. particle.wVec = particle.wVec * particle.dxVec * particle.dxVec  # wVec = dx.^2 .* wVec
            A11 = sum(particle.wVec)
            particle.wVec .= particle.wVec .* particle.dxVec  # wVec = dx.^3 .* wVec
            A12 = sum(particle.wVec)/2
            A22 = dot(particle.wVec, particle.dxVec)/4
            D = A11*A22 - A12^2

            particle.wVec .= muscl.weightFunction(particle.dxVec; param=interpAlpha, normalisation=particleGrid.dx)
            for i in eachindex(particle.alfaijBar)
                particle.alfaijBar[i] = (A22*particle.wVec[i]*particle.dxVec[i] - 0.5*A12*particle.wVec[i]*(particle.dxVec[i]^2))/D  # Derivative
                particle.betaij[i] = (0.5*A11*particle.wVec[i]*(particle.dxVec[i]^2) - A12*particle.wVec[i]*particle.dxVec[i])/D  # Second derivative
            end
        elseif ORDER == MUSCLORDER3
            @. particle.A[:, 1] = particle.dxVec * particle.wVec 
            @. particle.A[:, 2] = (particle.dxVec^2) * particle.wVec / 2
            @. particle.A[:, 3] = (particle.dxVec^3) * particle.wVec / 6

            coeff = pinv(particle.A[:, 1:3]; rtol=sqrt(eps(real(float(oneunit(eltype(particle.A)))))))
            @. particle.alfaijBar = coeff[1, :] * particle.wVec  # Derivative
            @. particle.betaij = coeff[2, :] * particle.wVec  # Second derivative
            @. particle.alfaij = coeff[3, :] * particle.wVec  # Third derivative
        elseif ORDER == MUSCLORDER4
            particle.wVec .= muscl.weightFunction(particle.dxVec; param=interpAlpha, normalisation=1.0)
            @. particle.A[:, 1] = particle.dxVec * particle.wVec 
            @. particle.A[:, 2] = (particle.dxVec^2) * particle.wVec / 2
            @. particle.A[:, 3] = (particle.dxVec^3) * particle.wVec / 6
            @. particle.A[:, 4] = (particle.dxVec^4) * particle.wVec / 24

            coeff = pinv(particle.A; rtol=sqrt(eps(real(float(oneunit(eltype(particle.A)))))))
            @. particle.alfaijBar = coeff[1, :] * particle.wVec  # Derivative
            @. particle.betaij = coeff[2, :] * particle.wVec  # Second derivative
            @. particle.alfaij = coeff[3, :] * particle.wVec  # Third derivative
            @. particle.gammaij = coeff[4, :] * particle.wVec  # Fourth derivative
        end
    end
end

function (muscl::MUSCL{ORDER})(particleGrid::ParticleGrid1D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::ScalarHyperbolicEquation{D}, settings::SimSetting; setCurvature::Bool=true)::Real where {ORDER<:MUSCLORDER, D}
    particle = particleGrid.grid[particleIndex]
    div = 0.0
    for (index, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
        deltaPos = getDistance(particleGrid, particleIndex, nbIndex)
        nbParticle = particleGrid.grid[nbIndex]

        if ORDER == MUSCLORDER1
            # Linear reconstruction from particleIndex and neighbour at center point 
            fij = fVec[particleIndex] + 0.5*deltaPos*sum(particle.alfaij[i]*(fVec[k] - fVec[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
            fji = fVec[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaij[i]*(fVec[k] - fVec[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
            fm, fp = sortFlux(fij, fji, deltaPos)
            div += particle.alfaij[index]*(muscl.numericalFlux(fm, fp, eq) - flux(eq, fVec[particleIndex]))
        elseif ORDER == MUSCLORDER2
            # Quadratic reconstruction
            fij = fVec[particleIndex]
            for (i, k) in enumerate(particle.neighbourIndices)
                fij += (deltaPos*particle.alfaijBar[i]/2 + (deltaPos^2)*particle.betaij[i]/8)*(fVec[k] - fVec[particleIndex])
            end
            fji = fVec[nbIndex]
            for (i, k) in enumerate(nbParticle.neighbourIndices)
                fji += (-deltaPos*nbParticle.alfaijBar[i]/2 + (deltaPos^2)*nbParticle.betaij[i]/8)*(fVec[k] - fVec[nbIndex])
            end
            fm, fp = sortFlux(fij, fji, deltaPos)
            div += particle.alfaijBar[index]*(muscl.numericalFlux(fm, fp, eq) - flux(eq, fVec[particleIndex]))
        elseif ORDER == MUSCLORDER3
            # Cubic reconstruction
            fij = fVec[particleIndex]
            for (i, k) in enumerate(particle.neighbourIndices)
                fij += (deltaPos*particle.alfaijBar[i]/2 + (deltaPos^2)*particle.betaij[i]/8 + (deltaPos^3)*particle.alfaij[i]/(6*8))*(fVec[k] - fVec[particleIndex])
            end
            fji = fVec[nbIndex]
            for (i, k) in enumerate(nbParticle.neighbourIndices)
                fji += (-deltaPos*nbParticle.alfaijBar[i]/2 + (deltaPos^2)*nbParticle.betaij[i]/8 - (deltaPos^3)*nbParticle.alfaij[i]/(6*8))*(fVec[k] - fVec[nbIndex])
            end
            fm, fp = sortFlux(fij, fji, deltaPos)
            div += particle.alfaijBar[index]*(muscl.numericalFlux(fm, fp, eq) - flux(eq, fVec[particleIndex]))
        elseif ORDER == MUSCLORDER4
            # Quartic reconstruction
            fij = fVec[particleIndex]
            for (i, k) in enumerate(particle.neighbourIndices)
                fij += (deltaPos*particle.alfaijBar[i]/2 + (deltaPos^2)*particle.betaij[i]/8 + (deltaPos^3)*particle.alfaij[i]/(6*8) + (deltaPos^4)*particle.gammaij[i]/(24*(2^4)))*(fVec[k] - fVec[particleIndex])
            end
            fji = fVec[nbIndex]
            for (i, k) in enumerate(nbParticle.neighbourIndices)
                fji += (-deltaPos*nbParticle.alfaijBar[i]/2 + (deltaPos^2)*nbParticle.betaij[i]/8 - (deltaPos^3)*nbParticle.alfaij[i]/(6*8) + (deltaPos^4)*nbParticle.gammaij[i]/(24*(2^4)))*(fVec[k] - fVec[nbIndex])
            end
            fm, fp = sortFlux(fij, fji, deltaPos)
            div += particle.alfaijBar[index]*(muscl.numericalFlux(fm, fp, eq) - flux(eq, fVec[particleIndex]))
        end
    end

    # set curvature
    if setCurvature && (ORDER == MUSCLORDER1)
        particle.curvature = 0.0
    elseif setCurvature
        particle.curvature = sum(particle.betaij[i]*(fVec[nbIndex] - fVec[particleIndex]) for (i, nbIndex) in enumerate(particle.neighbourIndices))  # Central difference for second-derivative
    end
    return 2*div # Minus sign in front of the divergence taken into account in the time stepper routine
end

function initTimeStep(muscl::MUSCL{ORDER}, particleGrid::ParticleGrid2D, interpAlpha::Real, interpRange::Real) where {ORDER<:MUSCLORDER}
    # Compute reconstruction for particle in each neighbourhood and store gradient coefficients
    for (particleIndex, particle) in enumerate(particleGrid.grid)

        for (i, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
            deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
            particle.dxVec[i] = deltaX
            particle.dyVec[i] = deltaY
        end
        particle.wVec .= muscl.weightFunction(particle.dxVec, particle.dyVec; param=interpAlpha, normalisation=interpRange)

        if ORDER == MUSCLORDER1
            A11 = A12 = A22 = 0.0
            for (w, dx) in zip(particle.wVec, particle.dxVec)
                A11 += w*(dx^2)
            end
            for (w, dy) in zip(particle.wVec, particle.dyVec)
                A22 += w*(dy^2)
            end
            for (w, dx, dy) in zip(particle.wVec, particle.dxVec, particle.dyVec)
                A12 += w*dx*dy
            end
            D = (A12^2) - A22*A11

            for i in 1:length(particle.dxVec)
                particle.alfaij[i] = (particle.wVec[i]*particle.dyVec[i]*A12 - A22*particle.wVec[i]*particle.dxVec[i])/D
                particle.betaij[i] = (-particle.wVec[i]*particle.dyVec[i]*A11 + A12*particle.wVec[i]*particle.dxVec[i])/D
            end    
        elseif ORDER == MUSCLORDER2
            @. particle.A[:, 1] = particle.dxVec * particle.wVec
            @. particle.A[:, 2] = particle.dyVec * particle.wVec
            @. particle.A[:, 3] = (particle.dxVec^2) * particle.wVec / 2
            @. particle.A[:, 4] = (particle.dyVec^2) * particle.wVec / 2
            @. particle.A[:, 5] = particle.dxVec * particle.dyVec * particle.wVec

            coeff = pinv(particle.A)
            particle.alfaij .= coeff[1, :] .* particle.wVec
            particle.betaij .= coeff[2, :] .* particle.wVec
            particle.alfaijBar .= coeff[3, :] .* particle.wVec
            particle.betaijBar .= coeff[4, :] .* particle.wVec
            particle.gammaij .= coeff[5, :] .* particle.wVec
        end
    end
end

function (muscl::MUSCL{ORDER})(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::ScalarHyperbolicEquation{D}, settings::SimSetting; setCurvature::Bool=true)::Real where {ORDER<:MUSCLORDER,D}
    particle = particleGrid.grid[particleIndex]
    div = 0.0
    for (index, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        nbParticle = particleGrid.grid[nbIndex]

        if ORDER == MUSCLORDER1  # Linear reconstruction from particleIndex and neighbour at center point 
            fij = fVec[particleIndex]
            for (i, k) in enumerate(particle.neighbourIndices)
                @inbounds fij += (deltaX*particle.alfaij[i] + deltaY*particle.betaij[i])*(fVec[k] - fVec[particleIndex])/2
            end
            fji = fVec[nbIndex]
            for (i, k) in enumerate(nbParticle.neighbourIndices)
                @inbounds fji -= (deltaX*nbParticle.alfaij[i] + deltaY*nbParticle.betaij[i])*(fVec[k] - fVec[nbIndex])/2
            end
        elseif ORDER == MUSCLORDER2  # Quadratic reconstruction from particleIndex and neighbour at center point
            fij = fVec[particleIndex] 
            for (i, k) in enumerate(particle.neighbourIndices)
                @inbounds fij += (deltaX*particle.alfaij[i] + deltaY*particle.betaij[i])*(fVec[k] - fVec[particleIndex])/2
                @inbounds fij += ((deltaX^2)*particle.alfaijBar[i]/2 + (deltaY^2)*particle.betaijBar[i]/2 + deltaX*deltaY*particle.gammaij[i])*(fVec[k] - fVec[particleIndex])/4
            end
            fji = fVec[nbIndex]
            for (i, k) in enumerate(nbParticle.neighbourIndices)
                @inbounds fji -= (deltaX*nbParticle.alfaij[i] + deltaY*nbParticle.betaij[i])*(fVec[k] - fVec[nbIndex])/2
                @inbounds fji += ((deltaX^2)*nbParticle.alfaijBar[i]/2 + (deltaY^2)*nbParticle.betaijBar[i]/2 + deltaX*deltaY*nbParticle.gammaij[i])*(fVec[k] - fVec[nbIndex])/4
            end
        end
        fmx, fpx, fmy, fpy = sortFlux(fij, fji, deltaX, deltaY)
        fx, fy = flux(eq, fVec[particleIndex])
        div += particle.alfaij[index]*(muscl.numericalFlux(fmx, fpx, eq, 1) - fx) + particle.betaij[index]*(muscl.numericalFlux(fmy, fpy, eq, 2) - fy)
    end

    # set curvature
    if setCurvature && (muscl.order == 1)
        particle.curvature[1] = 0.0
        particle.curvature[2] = 0.0
    elseif setCurvature
        particle.curvature[1] = sum(particle.alfaijBar[i]*(fVec[nbIndex] - fVec[particleIndex]) for (i, nbIndex) in enumerate(particle.neighbourIndices))  # Central difference for second-derivative
        particle.curvature[1] = sum(particle.betaijBar[i]*(fVec[nbIndex] - fVec[particleIndex]) for (i, nbIndex) in enumerate(particle.neighbourIndices))  # Central difference for second-derivative
    end
    
    return 2*div # Minus sign in front of the divergence taken into account in the time stepper routine
end

"""
    setCurvatures!(particleGrid::ParticleGrid, settings::SimSetting)

Compute curvatures on whole grid using a central MLS method. Overwrites the particleGrid.temp vector.
"""
function setCurvatures!(particleGrid::ParticleGrid1D, settings::SimSetting)
    central = CentralGradient(2)
    eq = LinearAdvection(0.0)
    map!(particle -> particle.rho, particleGrid.temp, particleGrid.grid)
    for particleIndex in particleGrid.interior_indices
        central(particleGrid, particleIndex, particleGrid.temp, eq, settings)
    end
end

function setCurvatures!(particleGrid::ParticleGrid2D, settings::SimSetting)
    eq = LinearAdvection((0.0, 0.0))
    central = CentralGradient(2)
    for particleIndex in eachindex(particleGrid.grid)  # Use first column of particleGrid.temp as temporary
        particleGrid.temp[particleIndex, 1] = particleGrid.grid[particleIndex].rho
    end
    for particleIndex in eachindex(particleGrid.grid)
        central(particleGrid, particleIndex, @view(particleGrid.temp[:, 1]), eq, settings)
    end
end

# In Interpolations.jl module

# (Keep existing using statements and other interpolator code like UpwindGradient, MUSCL, etc.)
# ...

# --- 1. Define Limiter Strategies and Helper Functions ---

abstract type AbstractSlopeLimiter end

struct BarthJespersenLimiter <: AbstractSlopeLimiter end
struct VenkatakrishnanLimiter <: AbstractSlopeLimiter end
struct SuperbeeLimiter <: AbstractSlopeLimiter end
struct MinmodLimiter <: AbstractSlopeLimiter end
struct NoLimiter <: AbstractSlopeLimiter end

# --- Limiter "phi" functions (they compute the limiter coefficient) ---

function minmod_phi(r::Real)::Float64
    if r <= 0.0
        return 0.0
    else
        return min(1.0, r)
    end
end

function superbee_phi(r::Real)::Float64
    if r <= 0.0
        return 0.0
    else
        return max(min(1.0, 2.0 * r), min(2.0, r))
    end
end

function venkatakrishnan_psi(r::Real)::Float64
    if r <= 0.0
        return 0.0
    end
    return (r^2 + 2.0 * r) / (r^2 + r + 2.0)
end

# Helper to find closest left and right neighbors (for classical limiters)
function find_closest_lr_neighbors_1D(particleGrid::ParticleGrid1D, p_idx::Integer, fVec::AbstractVector{<:Real})
    particle_i = particleGrid.grid[p_idx]
    best_idx_L, val_L, dist_L = nothing, nothing, nothing
    best_idx_R, val_R, dist_R = nothing, nothing, nothing
    min_abs_dist_L, min_dist_R = Inf, Inf

    for nb_actual_idx in particle_i.neighbourIndices
        dx_ij = getDistance(particleGrid, p_idx, nb_actual_idx)
        if dx_ij > 1e-9
            if dx_ij < min_dist_R
                min_dist_R = dx_ij; best_idx_R = nb_actual_idx;
                val_R = fVec[nb_actual_idx]; dist_R = dx_ij;
            end
        elseif dx_ij < -1e-9
            abs_dx_ij = abs(dx_ij)
            if abs_dx_ij < min_abs_dist_L
                min_abs_dist_L = abs_dx_ij; best_idx_L = nb_actual_idx;
                val_L = fVec[nb_actual_idx]; dist_L = dx_ij;
            end
        end
    end
    return best_idx_L, val_L, dist_L, best_idx_R, val_R, dist_R
end


# --- 2. Modified MUSCLlimited Struct Definition ---

mutable struct MUSCLlimited{ORDER<:MUSCLORDER, L<:AbstractSlopeLimiter} <: GradientInterpolator
    order::ORDER
    limiter_strategy::L # Strategy object, e.g., SuperbeeLimiter()
    res::Vector{Float64}
    weightFunction::MLSWeightFunction
    numericalFlux::NumericalFluxFunction
    limited_slopes_cache::Vector{Float64}
    unlimited_slopes_cache::Vector{Float64}

    function MUSCLlimited(
        order::Int64; 
        limiter::L = SuperbeeLimiter(), # Default to Superbee
        weightFunction::MLSWeightFunction = exponentialWeightFunction(), 
        numericalFlux::NumericalFluxFunction = RusanovFlux()
    ) where {L <: AbstractSlopeLimiter}
        @assert (order == 1) "MUSCLlimited currently only supports order=1 (linear reconstruction)"
        limited_slopes_cache_init = Vector{Float64}(undef, 0)
        new{MUSCLORDER1, L}(MUSCLORDER1(), limiter, Vector{Float64}(undef, order), weightFunction, numericalFlux, limited_slopes_cache_init, limited_slopes_cache_init)
    end
end


# --- 3. Main initTimeStep for MUSCLlimited (Orchestrator) ---
function initTimeStep(
    muscl::MUSCLlimited, # Generic for any limiter strategy
    particleGrid::ParticleGrid1D, 
    interpAlpha::Real, 
    interpRange::Real;
    first_stage::Bool = true
)
    N_total_particles = length(particleGrid.grid)
    if length(muscl.limited_slopes_cache) != N_total_particles
        resize!(muscl.limited_slopes_cache, N_total_particles)
        resize!(muscl.unlimited_slopes_cache, N_total_particles)
    end

    # --- Part 1: Calculate geometric alfaij coefficients (unchanged) ---
    for (particleIndex_outer, particle_outer) in enumerate(particleGrid.grid)
        # ... (Your existing, correct logic for calculating alfaij) ...
        num_neighbors = length(particle_outer.neighbourIndices)
        if length(particle_outer.dxVec) != num_neighbors; resize!(particle_outer.dxVec, num_neighbors); end
        if length(particle_outer.wVec) != num_neighbors; resize!(particle_outer.wVec, num_neighbors); end
        if length(particle_outer.alfaij) != num_neighbors; resize!(particle_outer.alfaij, num_neighbors); end
        for (i, nbIndex) in enumerate(particle_outer.neighbourIndices); particle_outer.dxVec[i] = getDistance(particleGrid, particleIndex_outer, nbIndex); end
        try muscl.weightFunction(particle_outer.dxVec; param=interpAlpha, normalisation=particleGrid.dx, wVec_out=particle_outer.wVec)
        catch e; if isa(e, MethodError); particle_outer.wVec .= muscl.weightFunction(particle_outer.dxVec; param=interpAlpha, normalisation=particleGrid.dx); else; rethrow(e); end; end
        wVec_times_dx = particle_outer.wVec .* particle_outer.dxVec
        t_sum_denominator = dot(wVec_times_dx, particle_outer.dxVec)
        if abs(t_sum_denominator) < 1e-12; fill!(particle_outer.alfaij, 0.0); else; particle_outer.alfaij .= wVec_times_dx ./ t_sum_denominator; end
    end

    fVec_internal = [p.rho for p in particleGrid.grid]

    # --- Part 2: Calculate Unlimited Slopes (unchanged) ---
    for i in 1:N_total_particles
        particle_i = particleGrid.grid[i]
        ui = fVec_internal[i]
        slope = 0.0
        if !isempty(particle_i.alfaij)
            for (k_idx, nb_idx) in enumerate(particle_i.neighbourIndices)
                slope += particle_i.alfaij[k_idx] * (fVec_internal[nb_idx] - ui)
            end
        end
        muscl.unlimited_slopes_cache[i] = slope
    end
    
    #current_strategy = first_stage ? muscl.limiter_strategy : NoLimiter()
    # --- Part 3: Apply the selected limiting strategy (Dispatch!) ---
    limit_slope!(
        muscl.limited_slopes_cache, # Output cache
        muscl.limiter_strategy,     # The strategy object (e.g., SuperbeeLimiter())
        muscl.unlimited_slopes_cache,           # The slopes to be limited
        particleGrid,
        fVec_internal
    )
end

# --- 4. Dispatched `limit_slope!` Helper Functions ---

function limit_slope!(
    limited_slopes_cache::Vector{Float64},
    strategy::NoLimiter,
    unlimited_slopes::Vector{Float64},
    particleGrid::ParticleGrid1D,
    fVec::AbstractVector{<:Real}
)
    for i = eachindex(limited_slopes_cache)
        limited_slopes_cache[i] = unlimited_slopes[i]
    end
end

# Version for classical, ratio-based limiters (Superbee, Minmod)
function limit_slope!(
    limited_slopes_cache::Vector{Float64},
    strategy::Union{SuperbeeLimiter, MinmodLimiter},
    unlimited_slopes::Vector{Float64},
    particleGrid::ParticleGrid1D,
    fVec::AbstractVector{<:Real}
)
    N = length(fVec)
    for i in 1:N
        # Find closest left and right neighbors to calculate one-sided slopes
        _, val_L, dist_L_val, _, val_R, dist_R_val = find_closest_lr_neighbors_1D(particleGrid, i, fVec)

        slope_L_os = 0.0
        if !isnothing(val_L) && abs(dist_L_val) > 1e-12
            slope_L_os = (fVec[i] - val_L) / (-dist_L_val)
        end
        
        slope_R_os = 0.0
        if !isnothing(val_R) && abs(dist_R_val) > 1e-12
            slope_R_os = (val_R - fVec[i]) / dist_R_val
        end

        # --- CORRECTED LOGIC ---

        # If the one-sided slopes have opposite signs, it indicates a local
        # extremum, and the limited slope should be zero to prevent oscillations.
        if slope_L_os * slope_R_os <= 0.0
            limited_slopes_cache[i] = 0.0
        else
            # Calculate the ratio r of the "upwind" to "downwind" slope.
            # Avoid division by zero.
            r_val = abs(slope_R_os) > 1e-12 ? slope_L_os / slope_R_os : 1.0

            # Calculate the limiter function phi(r) based on the strategy.
            phi = 0.0
            if strategy isa SuperbeeLimiter
                phi = superbee_phi(r_val)
            elseif strategy isa MinmodLimiter
                phi = minmod_phi(r_val)
            end
            
            # The limited slope is phi(r) times one of the one-sided slopes.
            # A common and robust choice is to use the right-sided slope.
            # This correctly applies both the minmod and superbee limiters.
            limited_slopes_cache[i] =  phi * slope_R_os
        end
    end
end


# Version for geometric, bound-based limiters (Barth-Jespersen, Venkatakrishnan)
function limit_slope!(
    limited_slopes_cache::Vector{Float64},
    strategy::Union{BarthJespersenLimiter, VenkatakrishnanLimiter},
    unlimited_slopes::Vector{Float64},
    particleGrid::ParticleGrid1D,
    fVec::AbstractVector{<:Real}
)
    N = length(fVec)
    for i in 1:N
        particle_i = particleGrid.grid[i]
        ui = fVec[i]
        sigma_i_unlimited = unlimited_slopes[i]
        
        if isempty(particle_i.neighbourIndices) || abs(sigma_i_unlimited) < 1e-12
            limited_slopes_cache[i] = 0.0
            continue
        end

        u_max_stencil, u_min_stencil = ui, ui
        for nb_idx in particle_i.neighbourIndices
            u_max_stencil = max(u_max_stencil, fVec[nb_idx])
            u_min_stencil = min(u_min_stencil, fVec[nb_idx])
        end

        phi_i = 1.0
        for nb_idx in particle_i.neighbourIndices
            dx_ij = getDistance(particleGrid, i, nb_idx)
            delta_recon = sigma_i_unlimited * dx_ij 
            
            if abs(delta_recon) < 1e-12; continue; end
            
            local phi_j::Float64
            if delta_recon > 0.0 # Overshoot
                delta_allowed = u_max_stencil - ui
                r = delta_allowed < 1e-12 ? 0.0 : delta_allowed / delta_recon
            else # Undershoot
                delta_allowed = u_min_stencil - ui
                r = delta_allowed > -1e-12 ? 0.0 : delta_allowed / delta_recon
            end

            if strategy isa BarthJespersenLimiter
                phi_j = min(1.0, r)
            elseif strategy isa VenkatakrishnanLimiter
                phi_j = venkatakrishnan_psi(r)
            end
            phi_i = min(phi_i, phi_j)
        end
        limited_slopes_cache[i] = clamp(phi_i, 0.0, 1.0) * sigma_i_unlimited
    end
end

# --- Functor for MUSCLlimited{MUSCLORDER1} ---
# This remains unchanged. It correctly uses the pre-calculated limited slopes.
function (muscl::MUSCLlimited{MUSCLORDER1})(
    particleGrid::ParticleGrid1D, 
    particleIndex::Integer, 
    fVec::AbstractVector{<:Real}, 
    eq::ScalarHyperbolicEquation{D}, 
    settings::SimSetting; 
    setCurvature::Bool=true,
)::Real where {D}
    # ... (code as you provided, it is correct) ...
    particle_i_data = particleGrid.grid[particleIndex]
    ui = fVec[particleIndex]
    div_val = 0.0
    sigma_i_lim = muscl.limited_slopes_cache[particleIndex]
    if setCurvature; particle_i_data.curvature = 0.0; end

    for (idx_in_stencil, actual_nb_idx) in enumerate(particle_i_data.neighbourIndices)
        deltaPos_ij = getDistance(particleGrid, particleIndex, actual_nb_idx)
        uj = fVec[actual_nb_idx]
        sigma_j_lim = muscl.limited_slopes_cache[actual_nb_idx]
        fij = ui + (deltaPos_ij / 2.0) * sigma_i_lim
        fji = uj - (deltaPos_ij / 2.0) * sigma_j_lim
        fm, fp = sortFlux(fij, fji, deltaPos_ij)
        num_flux_ij = muscl.numericalFlux(fm, fp, eq)
        div_val += particle_i_data.alfaij[idx_in_stencil] * (num_flux_ij - flux(eq, ui))
    end
    
    return 2.0 * div_val 
end
# ... (rest of your Interpolations.jl module) ...


# # --- NEW: MUSCLlimited specific code ---

# # --- Add Helper Functions (Place these near the top or within the new section) ---

# # Helper function to find closest left and right neighbors and their data
# # Returns: (idx_L, val_L, dist_L, idx_R, val_R, dist_R)
# # dist_L = (xj - xi) < 0, dist_R = (xk - xi) > 0
# function find_closest_lr_neighbors_1D(particleGrid::ParticleGrid1D, p_idx::Integer, fVec::Vector{<:Real})
#     # p_idx is the index of the central particle in particleGrid.grid
#     particle_i = particleGrid.grid[p_idx]

#     best_idx_L::Union{Int, Nothing} = nothing
#     val_L::Union{Float64, Nothing} = nothing
#     dist_L::Union{Float64, Nothing} = nothing # Will be < 0

#     best_idx_R::Union{Int, Nothing} = nothing
#     val_R::Union{Float64, Nothing} = nothing
#     dist_R::Union{Float64, Nothing} = nothing # Will be > 0

#     min_abs_dist_L = Inf
#     min_dist_R = Inf

#     # Iterate through the pre-identified neighbors of particle 'p_idx'
#     # neighbourIndices should contain the actual grid indices
#     for nb_actual_idx in particle_i.neighbourIndices
#         dx_ij = getDistance(particleGrid, p_idx, nb_actual_idx)
#         if dx_ij > 1e-9 # Potential right neighbor
#             if dx_ij < min_dist_R
#                 min_dist_R = dx_ij
#                 best_idx_R = nb_actual_idx
#                 val_R = fVec[nb_actual_idx]
#                 dist_R = dx_ij
#             end
#         elseif dx_ij < -1e-9 # Potential left neighbor
#             abs_dx_ij = abs(dx_ij)
#             if abs_dx_ij < min_abs_dist_L
#                 min_abs_dist_L = abs_dx_ij
#                 best_idx_L = nb_actual_idx
#                 val_L = fVec[nb_actual_idx]
#                 dist_L = dx_ij # Keep its negative sign
#             end
#         end
#     end
#     return best_idx_L, val_L, dist_L, best_idx_R, val_R, dist_R
# end

# # Superbee limiter function phi(r)
# function superbee_phi(r::Real)::Float64
#     if r <= 0.0
#         return 0.0
#     else
#         return max(min(1.0, 2.0 * r), min(2.0, r))
#     end
# end

# # --- New MUSCLlimited Struct Definition ---
# # (Keep the original MUSCL struct and its methods)

# # Using MUSCLORDER1 from original MUSCL definition
# mutable struct MUSCLlimited{ORDER<:MUSCLORDER} <: GradientInterpolator
#     order::ORDER # Will likely always be MUSCLORDER1 for this implementation
#     res::Vector{Float64}  # Result of gradient computation (size matches order)
#     weightFunction::MLSWeightFunction
#     numericalFlux::NumericalFluxFunction
#     limited_slopes_cache::Vector{Float64} # Cache for slopes limited in initTimeStep

#     # Constructor for MUSCLlimited - currently only supporting order 1 (linear reconstruction)
#     function MUSCLlimited(order::Int64; weightFunction::MLSWeightFunction = exponentialWeightFunction(), numericalFlux::NumericalFluxFunction = RusanovFlux())
#         @assert (order == 1) "MUSCLlimited currently only supports order=1 (linear reconstruction)"
#         # Need to know the number of particles N to initialize cache, but it's not available here.
#         # Initialize with size 0 and resize in initTimeStep.
#         limited_slopes_cache_init = Vector{Float64}(undef, 0)
#         # res size for order 1 is 1 (for 1D)
#         new{MUSCLORDER1}(MUSCLORDER1(), Vector{Float64}(undef, order), weightFunction, numericalFlux, limited_slopes_cache_init)
#     end
# end


# # --- initTimeStep specifically for MUSCLlimited{MUSCLORDER1} ---
# # MODIFIED: fVec argument is removed; it's constructed internally from particleGrid.
# function initTimeStep(
#     muscl::MUSCLlimited{MUSCLORDER1}, 
#     particleGrid::ParticleGrid1D, 
#     interpAlpha::Real, 
#     interpRange::Real
# )
#     N_total_particles = length(particleGrid.grid)
#     # Ensure cache is correctly sized
#     if length(muscl.limited_slopes_cache) != N_total_particles
#         resize!(muscl.limited_slopes_cache, N_total_particles)
#     end

#     # --- Part 1: Calculate original alfaij coefficients (needed for divergence sum) ---
#     # This part is geometric and depends only on particleGrid positions and weights.
#     for (particleIndex_outer, particle_outer) in enumerate(particleGrid.grid)
#         current_neighbors = particle_outer.neighbourIndices
#         num_neighbors = length(current_neighbors)

#         # Ensure internal vectors are sized (should be handled by updateNeighbours!)
#         if length(particle_outer.dxVec) != num_neighbors resize!(particle_outer.dxVec, num_neighbors) end
#         if length(particle_outer.wVec) != num_neighbors resize!(particle_outer.wVec, num_neighbors) end
#         if length(particle_outer.alfaij) != num_neighbors resize!(particle_outer.alfaij, num_neighbors) end

#         for (i, nbIndex) in enumerate(current_neighbors)
#             particle_outer.dxVec[i] = getDistance(particleGrid, particleIndex_outer, nbIndex)
#         end
        
#         # Calculate weights (wVec)
#         # Assuming muscl.weightFunction can write to an output vector or returns a new one.
#         # If it has a wVec_out keyword:
#         try
#              muscl.weightFunction(particle_outer.dxVec; param=interpAlpha, normalisation=particleGrid.dx, wVec_out=particle_outer.wVec)
#         catch e
#              if isa(e, MethodError) # Fallback if wVec_out is not implemented
#                  particle_outer.wVec .= muscl.weightFunction(particle_outer.dxVec; param=interpAlpha, normalisation=particleGrid.dx)
#              else
#                  rethrow(e)
#              end
#         end

#         # Calculate alfaij
#         wVec_times_dx = similar(particle_outer.dxVec) # Avoid modifying particle_outer.wVec if used later
#         for i in eachindex(wVec_times_dx)
#             wVec_times_dx[i] = particle_outer.wVec[i] * particle_outer.dxVec[i]
#         end
#         t_sum_denominator = dot(wVec_times_dx, particle_outer.dxVec)

#         if abs(t_sum_denominator) < 1e-12
#             fill!(particle_outer.alfaij, 0.0)
#         else
#             for i in eachindex(particle_outer.alfaij)
#                 particle_outer.alfaij[i] = wVec_times_dx[i] / t_sum_denominator
#             end
#         end
#     end

#     # --- Construct fVec internally from the current state in particleGrid ---
#     fVec_internal = Vector{Float64}(undef, N_total_particles)
#     for i in 1:N_total_particles
#         fVec_internal[i] = particleGrid.grid[i].rho
#     end

#     # --- Part 2: Calculate and store all limited slopes using fVec_internal ---
#     # Consider Threads.@threads for this loop if N_total_particles is large and functions are safe
#     for i in 1:N_total_particles
#         ui = fVec_internal[i] # Current particle's value from fVec_internal
        
#         # find_closest_lr_neighbors_1D needs the fVec_internal to get neighbor values
#         _, val_L, dist_L_val, _, val_R, dist_R_val = find_closest_lr_neighbors_1D(particleGrid, i, fVec_internal)

#         slope_L_os = 0.0
#         if !isnothing(val_L) && !isnothing(dist_L_val) && abs(dist_L_val) > 1e-9
#             slope_L_os = (ui - val_L) / (-dist_L_val) # (u_i - u_{i-1}) / (dx_i)
#         end

#         slope_R_os = 0.0
#         if !isnothing(val_R) && !isnothing(dist_R_val) && abs(dist_R_val) > 1e-9
#             slope_R_os = (val_R - ui) / dist_R_val # (u_{i+1} - u_i) / (dx_{i+1})
#         end

#         r_val = 0.0
#         if abs(slope_R_os) < 1e-12 # Denominator for r_val
#             # If both slopes are near zero, r_val=1 (phi=1, limited_slope=0).
#             # If only slope_R_os is zero, r_val=-1 (phi=0, limited_slope=0).
#             r_val = (abs(slope_L_os) < 1e-12) ? 1.0 : -1.0 
#         else
#             r_val = slope_L_os / slope_R_os
#         end

#         phi = superbee_phi(r_val)

#         # Apply limiter: phi * one_sided_slope (typically the "downwind" one if r=up/down)
#         # Or, if slopes have different signs (r<=0), limited slope is 0.
#         if slope_L_os * slope_R_os <= 1e-12 # If signs differ or one is zero, no overshoot from this form
#             muscl.limited_slopes_cache[i] = 0.0
#         else
#             # Superbee often applied as phi(r) * slope_R (if r = slope_L/slope_R)
#             # or minmod(slope_L, slope_R) if phi is minmod(1,r)
#             # For Superbee: phi(r) * slope_R ensures that if r > 2, it uses 2*slope_R, if r < 0.5, it uses 2r*slope_R = 2*slope_L
#             muscl.limited_slopes_cache[i] = phi * slope_R_os # This is a common way
#         end
#     end
# end


# # --- Functor for MUSCLlimited{MUSCLORDER1} ---
# # This uses the pre-calculated limited slopes from its cache
# function (muscl::MUSCLlimited{MUSCLORDER1})(particleGrid::ParticleGrid1D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::ScalarHyperbolicEquation, settings::SimSetting; setCurvature::Bool=true)::Real

#     # Note: This function *assumes* that `initTimeStep(muscl, particleGrid, ..., fVec)`
#     # has already been called for the relevant `fVec` stage, populating
#     # `muscl.limited_slopes_cache` and `particle.alfaij`.

#     particle_i_data = particleGrid.grid[particleIndex]
#     ui = fVec[particleIndex]
#     div_val = 0.0

#     # Retrieve the limited slope for the current particle 'particleIndex'
#     sigma_i_lim = muscl.limited_slopes_cache[particleIndex]

#     if setCurvature # For MUSCLORDER1, curvature is conceptually zero
#         particle_i_data.curvature = 0.0
#     end

#     for (idx_in_stencil, actual_nb_idx) in enumerate(particle_i_data.neighbourIndices)
#         deltaPos_ij = getDistance(particleGrid, particleIndex, actual_nb_idx)
#         uj = fVec[actual_nb_idx]

#         # Retrieve the limited slope for the neighbor particle 'actual_nb_idx'
#         sigma_j_lim = muscl.limited_slopes_cache[actual_nb_idx]

#         # Reconstruct states at midpoint using LIMITED slopes
#         fij = ui + (deltaPos_ij / 2.0) * sigma_i_lim
#         fji = uj - (deltaPos_ij / 2.0) * sigma_j_lim

#         # Numerical Flux
#         fm, fp = sortFlux(fij, fji, deltaPos_ij)
#         num_flux_ij = muscl.numericalFlux(fm, fp, eq)

#         # Accumulate divergence using original particle_i_data.alfaij coefficients
#         div_val += particle_i_data.alfaij[idx_in_stencil] * (num_flux_ij - flux(eq, ui))
#     end

#     return 2.0 * div_val # Factor of 2 as in original code
# end

# --- Keep the original initTimeStep and functor for the standard MUSCL struct ---
# function initTimeStep(muscl::MUSCL{ORDER}, ...) where {ORDER <: MUSCLORDER} ...
# function (muscl::MUSCL{ORDER})(...) where {ORDER <: MUSCLORDER} ...

# (Keep other interpolators like UpwindGradient, CentralGradient etc.)
# ...

end # End Module Interpolations