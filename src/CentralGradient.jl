# --- In Interpolations.jl ---

#==============================================================================
  CENTRAL GRADIENT (Unified 1D/2D - Optimized for SoA & Workspace)
==============================================================================#

# --- 1. Define Workspaces and Parametric Struct ---

abstract type CentralGradientWorkspace end

struct CentralGradientWorkspace1D <: CentralGradientWorkspace
    dx_buffer::Vector{Float64}
    df_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    function CentralGradientWorkspace1D(cap=20)
        new(zeros(cap), zeros(cap), zeros(cap))
    end
end

struct CentralGradientWorkspace2D <: CentralGradientWorkspace
    dx_buffer::Vector{Float64}
    dy_buffer::Vector{Float64}
    df_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    function CentralGradientWorkspace2D(cap=40)
        new(zeros(cap), zeros(cap), zeros(cap), zeros(cap))
    end
end

function ensure_capacity!(ws::CentralGradientWorkspace, n::Int)
    if n > length(ws.dx_buffer)
        N = n + n ÷ 4
        resize!(ws.dx_buffer, N); resize!(ws.df_buffer, N); resize!(ws.w_buffer, N)
        if ws isa CentralGradientWorkspace2D
            resize!(ws.dy_buffer, N)
        end
    end
end

mutable struct CentralGradient{D, WS <: CentralGradientWorkspace, I <: Interpolator} <: GradientInterpolator
    order::Int
    weightFunction::MLSWeightFunction
    workspace::WS
    interpolator::I

    function CentralGradient(order::Int, dimension::Int; weightFunction=exponentialWeightFunction())
        @assert order >= 1 "Order must be 1 or greater."
        ws = dimension == 1 ? CentralGradientWorkspace1D() : CentralGradientWorkspace2D()
        interpolator = Interpolator{dimension, order, 1}()
        WS = typeof(ws)
        I = typeof(interpolator)
        new{dimension, WS, I}(order, weightFunction, ws, interpolator)
    end
end

# --- 2. Refactored CentralGradient Functors (Dispatched for 1D and 2D) ---

function (central::CentralGradient{1})(
    particleGrid::ParticleGrid1D, 
    particleIndex::Integer, 
    fVec::AbstractVector, 
    eq::LinearAdvection{1}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    ws = central.workspace
    interp = central.interpolator
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    if num_neighbors < central.order; return 0.0; end

    ensure_capacity!(ws, num_neighbors)
    ensure_capacity!(interp, num_neighbors)
    dxVec = @view ws.dx_buffer[1:num_neighbors]
    dfVec = @view ws.df_buffer[1:num_neighbors]
    wVec  = @view ws.w_buffer[1:num_neighbors]

    for (i, nbIndex) in enumerate(neighbors)
        dxVec[i] = getDistance(particleGrid, particleIndex, nbIndex) / particleGrid.dx
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end
    central.weightFunction(wVec, dxVec; param=settings.interpAlpha, normalisation=1.0)

    res1, res2 = interp(dxVec, wVec, dfVec)

    if setCurvature
        if central.order == 1
            particleGrid.curvatures[particleIndex] = 0.0
        else # order == 2
            particleGrid.curvatures[particleIndex] = res2 / (particleGrid.dx^2)
        end
    end

    return velocity(eq, 0.0) * res1 / particleGrid.dx
end


function (central::CentralGradient{2})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    ws = central.workspace
    interp = central.interpolator
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    if num_neighbors < central.order; return 0.0; end

    ensure_capacity!(ws, num_neighbors)
    ensure_capacity!(interp, num_neighbors)
    dxVec = @view ws.dx_buffer[1:num_neighbors]
    dyVec = @view ws.dy_buffer[1:num_neighbors]
    dfVec = @view ws.df_buffer[1:num_neighbors]
    wVec  = @view ws.w_buffer[1:num_neighbors]
    
    norm_factor = max(particleGrid.dx, particleGrid.dy)

    for (i, nbIndex) in enumerate(neighbors)
        dx, dy = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = dx / norm_factor
        dyVec[i] = dy / norm_factor
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end
    central.weightFunction(wVec, dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)

    res1, res2, res3, res4 = interp(dxVec, dyVec, wVec, dfVec)
    
    if setCurvature
        if central.order == 1
            particleGrid.curvatures[particleIndex, :] .= 0.0
        else # order == 2
            particleGrid.curvatures[particleIndex, 1] = res3 / (norm_factor^2)
            particleGrid.curvatures[particleIndex, 2] = res4 / (norm_factor^2)
        end
    end
    
    vel = velocity(eq, 0.0)
    return (vel[1] * res1 + vel[2] * res2) / norm_factor
end

