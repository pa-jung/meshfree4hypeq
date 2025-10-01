# --- In Interpolations.jl ---

#==============================================================================
  CENTRAL GRADIENT (Unified 1D/2D - Optimized for SoA & Workspace)
==============================================================================#

# --- 1. Define Workspaces and Parametric Struct ---

abstract type CentralGradientWorkspace end

mutable struct CentralGradientWorkspace1D <: CentralGradientWorkspace
    dx_buffer::Vector{Float64}
    df_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    max_neighbors::Int
    function CentralGradientWorkspace1D(cap=20)
        new(zeros(cap), zeros(cap), zeros(cap), cap)
    end
end

mutable struct CentralGradientWorkspace2D <: CentralGradientWorkspace
    dx_buffer::Vector{Float64}
    dy_buffer::Vector{Float64}
    df_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    max_neighbors::Int
    function CentralGradientWorkspace2D(cap=40)
        new(zeros(cap), zeros(cap), zeros(cap), zeros(cap), cap)
    end
end

function ensure_capacity!(ws::CentralGradientWorkspace, n::Int)
    if n > ws.max_neighbors
        ws.max_neighbors = n
        resize!(ws.dx_buffer, n); resize!(ws.df_buffer, n); resize!(ws.w_buffer, n)
        if ws isa CentralGradientWorkspace2D
            resize!(ws.dy_buffer, n)
        end
    end
end

mutable struct CentralGradient{D} <: GradientInterpolator
    order::Int
    res::Vector{Float64}
    weightFunction::MLSWeightFunction
    workspace::CentralGradientWorkspace

    function CentralGradient(order::Int, dimension::Int; weightFunction=exponentialWeightFunction())
        @assert order >= 1 "Order must be 1 or greater."
        ws = dimension == 1 ? CentralGradientWorkspace1D() : CentralGradientWorkspace2D()
        res_size = (dimension == 1) ? order : (order == 1 ? 2 : 5)
        new{dimension}(order, zeros(res_size), weightFunction, ws)
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
    
    ws = central.workspace::CentralGradientWorkspace1D
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    if num_neighbors < central.order; return 0.0; end

    ensure_capacity!(ws, num_neighbors)
    dxVec = @view ws.dx_buffer[1:num_neighbors]
    dfVec = @view ws.df_buffer[1:num_neighbors]
    wVec  = @view ws.w_buffer[1:num_neighbors]

    for (i, nbIndex) in enumerate(neighbors)
        dxVec[i] = getDistance(particleGrid, particleIndex, nbIndex) / particleGrid.dx
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end
    wVec .= central.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)

    gradInterpolation!(dxVec, wVec, dfVec, central.res; order=central.order)

    if setCurvature
        if central.order == 1
            particleGrid.curvatures[particleIndex] = 0.0
        else # order == 2
            particleGrid.curvatures[particleIndex] = central.res[2] / (particleGrid.dx^2)
        end
    end

    return velocity(eq, 0.0) * central.res[1] / particleGrid.dx
end


function (central::CentralGradient{2})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    ws = central.workspace::CentralGradientWorkspace2D
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    if num_neighbors < central.order; return 0.0; end

    ensure_capacity!(ws, num_neighbors)
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
    wVec .= central.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)

    gradInterpolation!(dxVec, dyVec, wVec, dfVec, central.res; order=central.order)
    
    if setCurvature
        if central.order == 1
            particleGrid.curvatures[particleIndex, :] .= 0.0
        else # order == 2
            particleGrid.curvatures[particleIndex, 1] = central.res[3] / (norm_factor^2)
            particleGrid.curvatures[particleIndex, 2] = central.res[4] / (norm_factor^2)
        end
    end
    
    vel = velocity(eq, 0.0)
    return (vel[1] * central.res[1] + vel[2] * central.res[2]) / norm_factor
end

