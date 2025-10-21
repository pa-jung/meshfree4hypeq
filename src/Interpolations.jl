module Interpolations

using LinearAlgebra
using Statistics
using Base.Threads
using ..Meshfree4ScalarEq.ParticleGrids
using ..Meshfree4ScalarEq.SimSettings
using ..Meshfree4ScalarEq.HyperbolicPDEs
using ..Meshfree4ScalarEq.FluxFunctions
using ..Meshfree4ScalarEq.MLSWeightFunctions

export functionInterpolation!, gradInterpolation!, setCurvatures!, GradientInterpolator, initTimeStep, UpwindGradient, CentralGradient, WENO, MUSCL, AxelMUSCL, DumbserWENO, getStencil, LaxFriedrichsGradient, MUSCLlimited,
       AbstractSlopeLimiter, BarthJespersenLimiter, VenkatakrishnanLimiter, SuperbeeLimiter, MinmodLimiter, NoLimiter, NoFallbackGrad, Interpolator, initGI!, initGIBuffers!

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
using LinearAlgebra # For dot, pinv

mutable struct Interpolator{D, IO, DO}
    # Buffers for the weighted least-squares problem
    A::Matrix{Float64}
    b::Vector{Float64}
    res::Vector{Float64}
    w_buffer::Vector{Float64}

    function Interpolator{D, IO, DO}(max_points::Int=30) where {D, IO, DO}
        # Determine number of coefficients from orders and dimension
        # This is a simplified mapping; you can make it more general
        num_coeffs = D == 1 ? IO - DO + 1 : 5#IO == 1 ? D : (D == 1 ? IO+1 : 5) # 5 for 2nd order 2D

        new{D, IO, DO}(
            Matrix{Float64}(undef, max_points, num_coeffs),
            Vector{Float64}(undef, max_points),
            Vector{Float64}(undef, num_coeffs),
            Vector{Float64}(undef, num_coeffs),
        )
    end
end

function ensure_capacity!(interp::Interpolator, n::Int)
    if n > length(interp.b)
        new_capacity = n + n ÷ 4
        num_coeffs = size(interp.A, 2)
        
        # Re-create the matricx with the new capacity
        interp.A = Matrix{Float64}(undef, new_capacity, num_coeffs,)
        
        # Resize the vectors
        resize!(interp.b, new_capacity)
        resize!(interp.w_buffer, new_capacity)
    end
    return nothing
end

function (interp::Interpolator{1, 0, 0})(
    wVec::AbstractVector{<:Real},
    fVec::AbstractVector{<:Real}
)
    # This calculation is already non-allocating.
    s = sum(wVec)
    if abs(s) < 1e-14
        interp.res[1] = 0.0
    else
        interp.res[1] = dot(wVec, fVec) / s
    end
    return
end

function (interp::Interpolator{1, 1, 0})(
    dxVec::AbstractVector{<:Real},
    wVec::AbstractVector{<:Real},
    fVec::AbstractVector{<:Real}
)
    w_buffer = interp.w_buffer
    b1 = dot(fVec, wVec)
    A11 = sum(wVec)
    w_buffer .= wVec .* dxVec # w_temp = dx .* w
    b2 = dot(fVec, w_buffer)
    A12 = sum(w_buffer)
    w_buffer .*= dxVec  # w_temp = dx.^2 .* w
    A22 = sum(w_buffer)

    # Direct migration of your original 2x2 solver logic
    # res[1] is c₀, res[2] is c₁
    if abs(A12) < 1e-14 || abs(A12 - A22*A11/A12) < 1e-14
        return (0.0, 0.0)
    else
        res1 = (b2 - A22*b1/A12)/(A12 - A22*A11/A12)
        return res1, (b1 - A11*res1)/A12
    end
end
function (interp::Interpolator{1, 2, 0})(
    dxVec::AbstractVector{<:Real},
    wVec::AbstractVector{<:Real},
    fVec::AbstractVector{<:Real}
)
    w_buffer = interp.w_buffer
    # Generate normal equations
    b1 = dot(wVec, fVec)
    A11 = sum(wVec)
    w_buffer .= wVec .* dxVec  # w_temp = dx .* w
    A12 = sum(w_buffer)
    b2 = dot(w_buffer, fVec)
    w_buffer .*= dxVec  # w_temp = dx.^2 .* w
    A22 = sum(w_buffer)
    A13 = A22 / 2
    b3 = dot(w_buffer, fVec) / 2
    w_buffer .*= dxVec  # w_temp = dx.^3 .* w
    A23 = sum(w_buffer) / 2
    A33 = dot(w_buffer, dxVec) / 4

    # Hardcoded solve of 3x3 LU method
    L21 = A12 / A11
    L31 = A13 / A11
    U22 = A22 - L21 * A12
    L32 = (A23 - L31 * A12) / U22
    U23 = A23 - L21 * A13
    U33 = A33 - L31 * A13 - L32 * U23
    y2 = b2 - L21 * b1
    y3 = b3 - L31 * b1 - L32 * y2
    res3 = y3 / U33
    res2 = (y2 - U23 * res3) / U22
    return (b1 - A12 * res2 - A13 * res3) / A11, res2, res3
end

# Gradient Interpolators
function (interp::Interpolator{1, 1, 1})(
    dxVec::AbstractVector{<:Real},
    wVec::AbstractVector{<:Real},
    dfVec::AbstractVector{<:Real}
)
    w_buffer = interp.w_buffer
    w_buffer .= wVec .* dxVec 
    b1 = dot(dfVec, w_buffer)
    A11 = dot(w_buffer, dxVec)
    
    if abs(A11) < 1e-14
        return 0.0
    else
        return b1 / A11
    end
end

function (interp::Interpolator{2, 1, 1})(
    dxVec::AbstractVector{<:Real},
    dyVec::AbstractVector{<:Real},
    wVec::AbstractVector{<:Real},
    dfVec::AbstractVector{<:Real}
)
    A11 = 0.0; A12 = 0.0; A22 = 0.0
    b1 = 0.0; b2 = 0.0

    @inbounds for i in eachindex(dxVec)
        w = wVec[i]
        dx = dxVec[i]
        dy = dyVec[i]
        df = dfVec[i]
        
        A11 += w * dx * dx
        A22 += w * dy * dy
        A12 += w * dx * dy
        b1 += w * dx * df
        b2 += w * dy * df
    end
    
    # Explicit solve of 2x2 linear system
    D = (A12^2) - A22 * A11
    if abs(D) < 1e-14
        return 0.0,0.0
    else
        res1 = (b2 * A12 - A22 * b1) / D
        return res1, (b2 - A12 * res1) / A22
    end
end

function (interp::Interpolator{1, 2, 1})(
    dxVec::AbstractVector{<:Real},
    wVec::AbstractVector{<:Real},
    dfVec::AbstractVector{<:Real}
)
    # Generate normal equations
    w_buffer = interp.w_buffer
    w_buffer .= wVec .* dxVec
    b2 = dot(w_buffer, dfVec)
    w_buffer .*= dxVec
    A11 = sum(w_buffer)
    b3 = dot(w_buffer, dfVec) / 2
    w_buffer .*= dxVec
    A12 = sum(w_buffer) / 2
    A22 = dot(w_buffer, dxVec) / 4

    # Explicit solve of 2x2 linear system
    D = (A12^2) - A22 * A11
    if abs(D) < 1e-14
        return 0.0, 0.0
    else
        res1 = (b3 * A12 - A22 * b2) / D
        return res1, (b3 - A12 * res1) / A22
    end
    
    return
end

# function (interp::Interpolator{2, 2, 1})(
#     dxVec::AbstractVector{<:Real},
#     dyVec::AbstractVector{<:Real},
#     wVec::AbstractVector{<:Real},
#     dfVec::AbstractVector{<:Real};
#     robust_svd_solve::Bool=false # Solver toggle
# )
#     num_points = length(dxVec)
#     A_view = @view interp.A[1:num_points, :]
#     b_view = @view interp.b[1:num_points]
#     sqrt_w_view = @view interp.sqrt_w[1:num_points]

#     # Build the weighted system
#     sqrt_w_view .= sqrt.(wVec)
#     b_view .= dfVec .* sqrt_w_view

#     # Basis functions for 2D gradient: x, y, x^2/2, y^2/2, xy
#     A_view[:, 1] .= dxVec .* sqrt_w_view
#     A_view[:, 2] .= dyVec .* sqrt_w_view
#     A_view[:, 3] .= (dxVec.^2 ./ 2) .* sqrt_w_view
#     A_view[:, 4] .= (dyVec.^2 ./ 2) .* sqrt_w_view
#     A_view[:, 5] .= (dxVec .* dyVec) .* sqrt_w_view

#     # Solve using the chosen method
#     if robust_svd_solve
#         return pinv(A_view, rtol = sqrt(eps(eltype(A_view)))) * b_view
#     else
#         qr_factors = qr!(A_view, NoPivot())
#         res = @view interp.sqrt_w[1:5] # Minimal size 5 is ensured by default
#         # 2. Solve the system in-place into the result buffer.
#         ldiv!(res, qr_factors, b_view)
#         return res
#     end
# end

function (interp::Interpolator{2, 2, 1})(
    dxVec::AbstractVector{<:Real},
    dyVec::AbstractVector{<:Real},
    wVec::AbstractVector{<:Real},
    dfVec::AbstractVector{<:Real}
)
    # --- Directly construct the 5x5 Normal Matrix and RHS (Unchanged) ---
    N = @view interp.A[1:5, 1:5]
    b = @view interp.b[1:5]
    fill!(N, 0.0)
    fill!(b, 0.0)
    # --- Directly construct the 5x5 Normal Matrix and RHS in a single loop ---
    # The basis vector for each point is [x, y, x²/2, y²/2, xy]
    @inbounds for i in 1:num_points
        w = wVec[i]
        dx = dxVec[i]
        dy = dyVec[i]
        df = dfVec[i]

        # Precompute basis functions for the i-th point
        basis_i = (dx, dy, dx^2/2, dy^2/2, dx*dy)
        
        # Update the right-hand side rhs = AᵀWb
        for j in 1:5
            rhs_vec[j] += w * basis_i[j] * df
        end
        
        # Update the upper triangle of the symmetric normal matrix N = AᵀWA
        for j in 1:5
            for k in j:5
                N_matrix[j, k] += w * basis_i[j] * basis_i[k]
            end
        end
    end
    
    # Fill in the lower triangle of the symmetric matrix
    for j in 2:5
        for k in 1:(j-1)
            N_matrix[j, k] = N_matrix[k, j]
        end
    end

    # --- Fully Hardcoded 5x5 Cholesky Solver ---
    # We use local variables for clarity and to help the compiler.
    # This block has zero allocations and no function call overhead.
    
    # 1. Cholesky Decomposition (N = LLᵀ), calculating L
    l11 = sqrt(N[1,1])
    if l11 < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    inv_l11 = 1.0 / l11
    l21 = N[2,1] * inv_l11
    l31 = N[3,1] * inv_l11
    l41 = N[4,1] * inv_l11
    l51 = N[5,1] * inv_l11

    l22 = sqrt(N[2,2] - l21*l21)
    if l22 < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    inv_l22 = 1.0 / l22
    l32 = (N[3,2] - l31*l21) * inv_l22
    l42 = (N[4,2] - l41*l21) * inv_l22
    l52 = (N[5,2] - l51*l21) * inv_l22

    l33 = sqrt(N[3,3] - l31*l31 - l32*l32)
    if l33 < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    inv_l33 = 1.0 / l33
    l43 = (N[4,3] - l41*l31 - l42*l32) * inv_l33
    l53 = (N[5,3] - l51*l31 - l52*l32) * inv_l33

    l44 = sqrt(N[4,4] - l41*l41 - l42*l42 - l43*l43)
    if l44 < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    inv_l44 = 1.0 / l44
    l54 = (N[5,4] - l51*l41 - l52*l42 - l53*l43) * inv_l44

    l55 = sqrt(N[5,5] - l51*l51 - l52*l52 - l53*l53 - l54*l54)
    if l55 < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    
    # 2. Forward Substitution (solves Ly = b for y)
    y1 = b[1] * inv_l11
    y2 = (b[2] - l21*y1) * inv_l22
    y3 = (b[3] - l31*y1 - l32*y2) * inv_l33
    y4 = (b[4] - l41*y1 - l42*y2 - l43*y3) * inv_l44
    y5 = (b[5] - l51*y1 - l52*y2 - l53*y3 - l54*y4) / l55

    # 3. Backward Substitution (solves Lᵀx = y for x)
    res5 = y5 / l55
    res4 = (y4 - l54*res5) / l44
    res3 = (y3 - l43*res4 - l53*res5) / l33
    res2 = (y2 - l32*res3 - l42*res4 - l52*res5) / l22
    res1 = (y1 - l21*res2 - l31*res3 - l41*res4 - l51*res5) / l11

    return (res1, res2, res3, res4, res5)
end

"""
    GradientInterpolator

In case of unstructured grids, the spatial gradient is approximated using a moving least squares (MLS) method based on Taylor polynomials.
These algorithms are implemented as follows. Each method is a struct that is a subtype of GradientInterpolator. The gradient 
at a gridpoint can then be computed using the ()-operator; see for example UpwindGradient and CentralGradient. These objects select
the correct stencil and then call the MLS routine (gradInterpolation).
"""
abstract type GradientInterpolator end

# Fallback Gradient interpolator for no fallback
struct NoFallbackGrad <: GradientInterpolator end

function initGI!(::NoFallbackGrad, kwargs...)
    return
end

function initGIBuffers!(::NoFallbackGrad, kwargs...)
    return
end

function initTimeStep(g::GradientInterpolator, particleGrid::ParticleGrid) end  # Function called at the start of a time step (order RK-stage)

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