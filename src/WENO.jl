
abstract type WENOWorkspace end

mutable struct WENOWorkspace1D <: WENOWorkspace
    dx_buffer::Vector{Float64}
    df_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    left_window_buffer::BitVector
    max_neighbors::Int

    function WENOWorkspace1D(;cap=20)
        new(zeros(cap), zeros(cap), zeros(cap), falses(cap), cap)
    end
end

mutable struct WENOWorkspace2D <: WENOWorkspace
    dx_buffer::Vector{Float64}
    dy_buffer::Vector{Float64}
    df_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    left_window_buffer::BitVector
    top_window_buffer::BitVector
    max_neighbors::Int
    
    function WENOWorkspace2D(;cap=40)
        new(zeros(cap), zeros(cap), zeros(cap), zeros(cap), falses(cap), falses(cap), cap)
    end
end

function ensure_capacity!(ws::WENOWorkspace, n::Int)
    if n > ws.max_neighbors
        ws.max_neighbors = n
        resize!(ws.dx_buffer, n); resize!(ws.df_buffer, n); resize!(ws.w_buffer, n)
        resize!(ws.left_window_buffer, n)
        if ws isa WENOWorkspace2D
            resize!(ws.dy_buffer, n)
            resize!(ws.top_window_buffer, n)
        end
    end
end

mutable struct WENO{D} <: GradientInterpolator
    order::Int
    res::Vector{Float64}
    weightFunction::MLSWeightFunction
    workspace::WENOWorkspace

    function WENO(order::Int, dimension::Int; weightFunction=exponentialWeightFunction())
        @assert order >= 2 "WENO requires order >= 2 for second derivatives."
        ws = dimension == 1 ? WENOWorkspace1D() : WENOWorkspace2D()
        res_size = (dimension == 1) ? order : (order == 1 ? 2 : 5)
        new{dimension}(order, zeros(res_size), weightFunction, ws)
    end
end

# --- 2. Refactored WENO Functors (Dispatched for 1D and 2D) ---

function (weno::WENO{1})(
    particleGrid::ParticleGrid1D, 
    particleIndex::Integer, 
    fVec::AbstractVector, 
    eq::LinearAdvection{1}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    ws = weno.workspace::WENOWorkspace1D
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    if num_neighbors < weno.order; return 0.0; end # Not enough points for interpolation

    ensure_capacity!(ws, num_neighbors)
    dxVec = @view ws.dx_buffer[1:num_neighbors]
    dfVec = @view ws.df_buffer[1:num_neighbors]
    wVec = @view ws.w_buffer[1:num_neighbors]
    leftWindow = @view ws.left_window_buffer[1:num_neighbors]

    for (i, nbIndex) in enumerate(neighbors)
        dxVec[i] = getDistance(particleGrid, particleIndex, nbIndex)
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        leftWindow[i] = dxVec[i] < 0.0
    end
    wVec .= weno.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)

    # One-sided stencil
    stencil = velocity(eq, 0.0) > 0.0 ? leftWindow : .!leftWindow
    if count(stencil) < weno.order; return 0.0; end # Not enough points in one-sided stencil
    
    gradInterpolation!(dxVec[stencil], wVec[stencil], dfVec[stencil], weno.res; order=weno.order)
    resS1, resS2 = weno.res[1], weno.res[2]

    # Central stencil
    gradInterpolation!(dxVec, wVec, dfVec, weno.res; order=weno.order)
    resC1, resC2 = weno.res[1], weno.res[2]

    # Non-linear weights
    e = 1e-6
    dx2 = particleGrid.dx^2; dx4 = dx2^2
    betaS = 0.5 / ((resS1^2 * dx2 + resS2^2 * dx4 + e)^2)
    betaC = 0.5 / ((resC1^2 * dx2 + resC2^2 * dx4 + e)^2)
    
    if (betaC + betaS) < 1e-14; return 0.0; end
    ω_s = betaS / (betaC + betaS)
    ω_c = betaC / (betaC + betaS)

    if setCurvature
        particleGrid.curvatures[particleIndex] = resS2*ω_s + resC2*ω_c
    end

    return (resS1*ω_s + resC1*ω_c) * velocity(eq, 0.0)
end


function (weno::WENO{2})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    ws = weno.workspace::WENOWorkspace2D
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    
    # A robust WENO scheme requires a minimum number of points
    if num_neighbors < weno.order
        if setCurvature; particleGrid.curvatures[particleIndex, :] .= 0.0; end
        return 0.0
    end

    ensure_capacity!(ws, num_neighbors)
    dxVec = @view ws.dx_buffer[1:num_neighbors]
    dyVec = @view ws.dy_buffer[1:num_neighbors]
    dfVec = @view ws.df_buffer[1:num_neighbors]
    wVec = @view ws.w_buffer[1:num_neighbors]
    leftWindow = @view ws.left_window_buffer[1:num_neighbors]
    topWindow = @view ws.top_window_buffer[1:num_neighbors]

    # --- Populate Buffers ---
    for (i, nbIndex) in enumerate(neighbors)
        dx, dy = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i], dyVec[i] = dx / settings.interpRange, dy / settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        leftWindow[i] = dx < 0.0
        topWindow[i] = dy > 0.0
    end
    wVec .= weno.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
    vel = velocity(eq, 0.0)

    # --- Stencil Calculations (Left/Right, Up/Down, Central) ---
    
    # Horizontal Stencil (Left/Right)
    stencil_h = vel[1] > 0.0 ? leftWindow : .!leftWindow
    if count(stencil_h) < weno.order; return 0.0; end # Fallback if stencil is too small
    gradInterpolation!((@view dxVec[stencil_h]), (@view dyVec[stencil_h]), (@view wVec[stencil_h]), (@view dfVec[stencil_h]), weno.res; order=weno.order)
    resHx = weno.res[1]/settings.interpRange
    resHy = weno.res[2]/settings.interpRange
    resHxx = weno.res[3]/(settings.interpRange^2)
    resHyy = weno.res[4]/(settings.interpRange^2)
    resHxy = weno.res[5]/(settings.interpRange^2)
    
    # Vertical Stencil (Up/Down)
    # Note: The weight vector `wVec` is the same for all stencils, no need to recalculate
    stencil_v = vel[2] < 0.0 ? topWindow : .!topWindow
    if count(stencil_v) < weno.order; return 0.0; end # Fallback if stencil is too small
    gradInterpolation!((@view dxVec[stencil_v]), (@view dyVec[stencil_v]), (@view wVec[stencil_v]), (@view dfVec[stencil_v]), weno.res; order=weno.order)
    resVx = weno.res[1]/settings.interpRange
    resVy = weno.res[2]/settings.interpRange
    resVxx = weno.res[3]/(settings.interpRange^2)
    resVyy = weno.res[4]/(settings.interpRange^2)
    resVxy = weno.res[5]/(settings.interpRange^2)

    # Central Stencil
    gradInterpolation!(dxVec, dyVec, wVec, dfVec, weno.res; order=weno.order)
    resCx = weno.res[1]/settings.interpRange
    resCy = weno.res[2]/settings.interpRange
    resCxx = weno.res[3]/(settings.interpRange^2)
    resCyy = weno.res[4]/(settings.interpRange^2)
    resCxy = weno.res[5]/(settings.interpRange^2)

    # --- Non-linear Weights ---
    e = 1e-12
    # Use dx for a stable length scale, not interpRange which is arbitrary
    dx2 = particleGrid.dx^2; dx4 = dx2^2 
    
    # Combine all smoothness indicators for robustness
    betaH = 0.5 / ((resHx^2 + resHy^2)*dx2 + (resHxx^2 + resHyy^2 + resHxy^2)*dx4 + e)^2
    betaV = 0.5 / ((resVx^2 + resVy^2)*dx2 + (resVxx^2 + resVyy^2 + resVxy^2)*dx4 + e)^2
    betaC = 0.5 / ((resCx^2 + resCy^2)*dx2 + (resCxx^2 + resCyy^2 + resCxy^2)*dx4 + e)^2
    
    # Avoid division by zero
    sum_beta_h = betaH + betaC; sum_beta_v = betaV + betaC;
    if sum_beta_h < 1e-14 || sum_beta_v < 1e-14; return 0.0; end

    wH = betaH/sum_beta_h; wCx = betaC/sum_beta_h
    wV = betaV/sum_beta_v; wCy = betaC/sum_beta_v

    if setCurvature
        particleGrid.curvatures[particleIndex, 1] = wH*resHxx + wCx*resCxx
        particleGrid.curvatures[particleIndex, 2] = wV*resVyy + wCy*resCyy
    end

    # Return the final weighted divergence
    return (wH*resHx + wCx*resCx)*vel[1] + (wV*resVy + wCy*resCy)*vel[2]
end