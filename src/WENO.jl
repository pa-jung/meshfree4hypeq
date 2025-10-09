
abstract type WENOWorkspace end

struct WENOWorkspace1D <: WENOWorkspace
    # Main buffers for all neighbors
    dx_buffer::Vector{Float64}
    df_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    left_window_buffer::BitVector
    
    # Scratch space for stencil calculations
    dx_scratch::Vector{Float64}
    df_scratch::Vector{Float64}
    w_scratch::Vector{Float64}
    

    function WENOWorkspace1D(max_neighbors::Int=30)
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            BitVector(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors)
        )
    end
end

struct WENOWorkspace2D <: WENOWorkspace
    # Main buffers
    dx_buffer::Vector{Float64}; dy_buffer::Vector{Float64}
    df_buffer::Vector{Float64}; w_buffer::Vector{Float64}
    left_window_buffer::BitVector; top_window_buffer::BitVector
    
    # Scratch space for stencil calculations
    dx_scratch::Vector{Float64}; dy_scratch::Vector{Float64}
    df_scratch::Vector{Float64}; w_scratch::Vector{Float64}

    function WENOWorkspace2D(max_neighbors::Int=30)
        new(
            Vector{Float64}(undef, max_neighbors), Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors), Vector{Float64}(undef, max_neighbors),
            BitVector(undef, max_neighbors), BitVector(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors), Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors), Vector{Float64}(undef, max_neighbors)
        )
    end
end

function ensure_capacity!(ws::WENOWorkspace1D, n::Int)
    if n > length(ws.dx_buffer)
        new_capacity = n + n ÷ 4
        resize!.((ws.dx_buffer, ws.df_buffer, ws.w_buffer, 
                  ws.dx_scratch, ws.df_scratch, ws.w_scratch), new_capacity)
        resize!(ws.left_window_buffer, new_capacity)
    end
    return nothing
end

function ensure_capacity!(ws::WENOWorkspace2D, n::Int)
    if n > length(ws.dx_buffer)
        new_capacity = n + n ÷ 4
        resize!.((ws.dx_buffer, ws.dy_buffer, ws.df_buffer, ws.w_buffer,
                  ws.dx_scratch, ws.dy_scratch, ws.df_scratch, ws.w_scratch), new_capacity)
        resize!.((ws.left_window_buffer, ws.top_window_buffer), new_capacity)
    end
    return nothing
end

struct WENO{D,WS <: WENOWorkspace, I <: Interpolator} <: GradientInterpolator
    order::Int
    weightFunction::MLSWeightFunction
    workspace::WS
    interpolator::I

    function WENO(order::Int, dimension::Int; weightFunction=exponentialWeightFunction())
        @assert order >= 2 "WENO requires order >= 2 for second derivatives."
        ws = dimension == 1 ? WENOWorkspace1D() : WENOWorkspace2D()
        interpolator = Interpolator{dimension, order, 1}() 
        WS = typeof(ws)
        I = typeof(interpolator)
        new{dimension,WS,I}(order, weightFunction, ws, interpolator)
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
    interp = weno.interpolator
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    if num_neighbors < weno.order; return 0.0; end # Not enough points for interpolation
    ensure_capacity!(ws, num_neighbors)
    ensure_capacity!(interp, num_neighbors)
    dxVec = @view ws.dx_buffer[1:num_neighbors]
    dfVec = @view ws.df_buffer[1:num_neighbors]
    wVec = @view ws.w_buffer[1:num_neighbors]
    leftWindow = @view ws.left_window_buffer[1:num_neighbors]

    for (i, nbIndex) in enumerate(neighbors)
        dxVec[i] = getDistance(particleGrid, particleIndex, nbIndex)
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        leftWindow[i] = dxVec[i] < 0.0
    end
    weno.weightFunction(wVec, dxVec; param=settings.interpAlpha, normalisation=1.0)

    stencil_size = 0
    if velocity(eq, 0.0) > 0.0 # Left-sided
        for i in 1:num_neighbors
            if leftWindow[i]
                stencil_size += 1
                ws.dx_scratch[stencil_size] = dxVec[i]
                ws.w_scratch[stencil_size] = wVec[i]
                ws.df_scratch[stencil_size] = dfVec[i]
            end
        end
    else # Right-sided
        for i in 1:num_neighbors
            if !leftWindow[i]
                stencil_size += 1
                ws.dx_scratch[stencil_size] = dxVec[i]
                ws.w_scratch[stencil_size] = wVec[i]
                ws.df_scratch[stencil_size] = dfVec[i]
            end
        end
    end

    if stencil_size < weno.order; return 0.0; end

    # Create views of the populated scratch buffers
    dx_stencil = @view ws.dx_scratch[1:stencil_size]
    w_stencil  = @view ws.w_scratch[1:stencil_size]
    df_stencil = @view ws.df_scratch[1:stencil_size]
    
    resS1, resS2 = interp(dx_stencil, w_stencil, df_stencil)

    # Central stencil (uses the original, unmodified buffers)
    
    resC1, resC2 = interp(dxVec, wVec, dfVec)
    

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
    interp = weno.interpolator
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    
    if num_neighbors < weno.order
        if setCurvature; particleGrid.curvatures[particleIndex, :] .= 0.0; end
        return 0.0
    end

    ensure_capacity!(ws, num_neighbors)
    ensure_capacity!(interp, num_neighbors)
    dxVec = @view ws.dx_buffer[1:num_neighbors]
    dyVec = @view ws.dy_buffer[1:num_neighbors]
    dfVec = @view ws.df_buffer[1:num_neighbors]
    wVec  = @view ws.w_buffer[1:num_neighbors]
    leftWindow = @view ws.left_window_buffer[1:num_neighbors]
    topWindow = @view ws.top_window_buffer[1:num_neighbors]

    # --- Populate Main Buffers ---
    for (i, nbIndex) in enumerate(neighbors)
        dx, dy = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i], dyVec[i] = dx / settings.interpRange, dy / settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        leftWindow[i] = dx < 0.0
        topWindow[i] = dy > 0.0
    end
    weno.weightFunction(wVec, dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
    vel = velocity(eq, 0.0)

    # --- Horizontal Stencil (Using Scratch Buffers) ---
    stencil_size_h = 0
    if vel[1] > 0.0 # Left-sided
        for i in 1:num_neighbors
            if leftWindow[i]
                stencil_size_h += 1
                ws.dx_scratch[stencil_size_h] = dxVec[i]
                ws.dy_scratch[stencil_size_h] = dyVec[i]
                ws.w_scratch[stencil_size_h] = wVec[i]
                ws.df_scratch[stencil_size_h] = dfVec[i]
            end
        end
    else # Right-sided
        for i in 1:num_neighbors
            if !leftWindow[i]
                stencil_size_h += 1
                ws.dx_scratch[stencil_size_h] = dxVec[i]
                ws.dy_scratch[stencil_size_h] = dyVec[i]
                ws.w_scratch[stencil_size_h] = wVec[i]
                ws.df_scratch[stencil_size_h] = dfVec[i]
            end
        end
    end

    if stencil_size_h < weno.order; return 0.0; end
    dx_stencil_h = @view ws.dx_scratch[1:stencil_size_h]
    dy_stencil_h = @view ws.dy_scratch[1:stencil_size_h]
    w_stencil_h  = @view ws.w_scratch[1:stencil_size_h]
    df_stencil_h = @view ws.df_scratch[1:stencil_size_h]
    
    resHx, resHy, resHxx, resHyy, resHxy = interp(dx_stencil_h, dy_stencil_h, w_stencil_h, df_stencil_h)
    range = settings.interpRange
    resHx /= range; resHy /= range; resHxx /= range^2; resHyy /= range^2; resHxy /= range^2
    # --- Vertical Stencil (Using Scratch Buffers) ---
    stencil_size_v = 0
    if vel[2] < 0.0 # Top-sided
        for i in 1:num_neighbors
            if topWindow[i]
                stencil_size_v += 1
                ws.dx_scratch[stencil_size_v] = dxVec[i]
                ws.dy_scratch[stencil_size_v] = dyVec[i]
                ws.w_scratch[stencil_size_v] = wVec[i]
                ws.df_scratch[stencil_size_v] = dfVec[i]
            end
        end
    else # Bottom-sided
        for i in 1:num_neighbors
            if !topWindow[i]
                stencil_size_v += 1
                ws.dx_scratch[stencil_size_v] = dxVec[i]
                ws.dy_scratch[stencil_size_v] = dyVec[i]
                ws.w_scratch[stencil_size_v] = wVec[i]
                ws.df_scratch[stencil_size_v] = dfVec[i]
            end
        end
    end

    if stencil_size_v < weno.order; return 0.0; end
    dx_stencil_v = @view ws.dx_scratch[1:stencil_size_v]
    dy_stencil_v = @view ws.dy_scratch[1:stencil_size_v]
    w_stencil_v  = @view ws.w_scratch[1:stencil_size_v]
    df_stencil_v = @view ws.df_scratch[1:stencil_size_v]
    interp(dx_stencil_v, dy_stencil_v, w_stencil_v, df_stencil_v)
    resVx, resVy, resVxx, resVyy, resVxy = interp.res[1]/settings.interpRange, interp.res[2]/settings.interpRange, interp.res[3]/(settings.interpRange^2), interp.res[4]/(settings.interpRange^2), interp.res[5]/(settings.interpRange^2)

    # --- Central Stencil (Uses the original, unmodified buffers) ---
    interp(dxVec, dyVec, wVec, dfVec)
    resCx, resCy, resCxx, resCyy, resCxy = interp.res[1]/settings.interpRange, interp.res[2]/settings.interpRange, interp.res[3]/(settings.interpRange^2), interp.res[4]/(settings.interpRange^2), interp.res[5]/(settings.interpRange^2)

    # --- Non-linear Weights (Logic is unchanged) ---
    e = 1e-12
    dx2 = particleGrid.dx^2; dx4 = dx2^2 
    betaH = 0.5 / ((resHx^2 + resHy^2)*dx2 + (resHxx^2 + resHyy^2 + resHxy^2)*dx4 + e)^2
    betaV = 0.5 / ((resVx^2 + resVy^2)*dx2 + (resVxx^2 + resVyy^2 + resVxy^2)*dx4 + e)^2
    betaC = 0.5 / ((resCx^2 + resCy^2)*dx2 + (resCxx^2 + resCyy^2 + resCxy^2)*dx4 + e)^2
    
    sum_beta_h = betaH + betaC; sum_beta_v = betaV + betaC;
    if sum_beta_h < 1e-14 || sum_beta_v < 1e-14; return 0.0; end

    wH = betaH/sum_beta_h; wCx = betaC/sum_beta_h
    wV = betaV/sum_beta_v; wCy = betaC/sum_beta_v

    if setCurvature
        particleGrid.curvatures[particleIndex, 1] = wH*resHxx + wCx*resCxx
        particleGrid.curvatures[particleIndex, 2] = wV*resVyy + wCy*resCyy
    end

    return (wH*resHx + wCx*resCx)*vel[1] + (wV*resVy + wCy*resCy)*vel[2]
end

# --- In your Interpolations.jl file ---

#==============================================================================
  Dumbser WENO Scheme (Optimized for SoA Grids & Workspace)
==============================================================================#

# --- 1. Define Helper and Workspace ---

function getStencil(deltaX::Real, deltaY::Real, s::Int)
    # This robust version maps the angle from atan to an integer sector [0, s-1]
    angle = atan(deltaY, deltaX)
    # Shift angle to be in [0, 2*pi]
    if angle < 0.0
        angle += 2.0 * pi
    end
    # Normalize to [0, s] and floor to get the integer index
    stencil = floor(Int, (angle * s) / (2.0 * pi))
    # Clamp to ensure it's in the range [0, s-1] due to floating point nuances
    return clamp(stencil, 0, s - 1)
end


mutable struct DumbserWENOWorkspace <: MUSCLWorkspace
    # Buffers for neighbor-specific calculations
    dx_buffer::Vector{Float64}
    dy_buffer::Vector{Float64}
    df_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    
    # Buffers specific to Dumbser WENO logic
    window_matrix::Matrix{Bool}
    gradients::Matrix{Float64}
    weights::Vector{Float64}
    
    max_neighbors::Int
    
    function DumbserWENOWorkspace(s::Int=8, initial_capacity::Int=40)
        new(
            zeros(initial_capacity), zeros(initial_capacity), zeros(initial_capacity),
            zeros(initial_capacity),
            falses(initial_capacity, s + 1), # s one-sided stencils + 1 central
            zeros(5, s + 1), # 5 derivatives (x, y, xx, yy, xy) for each stencil
            zeros(s + 1),
            initial_capacity
        )
    end
end

# Specialize ensure_capacity! for the new workspace
function ensure_capacity!(ws::DumbserWENOWorkspace, n::Int)
    if n > ws.max_neighbors
        ws.max_neighbors = n
        resize!(ws.dx_buffer, n); resize!(ws.dy_buffer, n);
        resize!(ws.df_buffer, n); resize!(ws.w_buffer, n);
        ws.window_matrix = falses(n, size(ws.window_matrix, 2))
    end
end

# --- 2. Refactored DumbserWENO Struct and Constructor ---

mutable struct DumbserWENO <: GradientInterpolator
    order::Int
    res::Vector{Float64}
    weightFunction::MLSWeightFunction
    s::Int # amount of one-sided stencils
    workspace::DumbserWENOWorkspace

    function DumbserWENO(order::Int=2; weightFunction::MLSWeightFunction=exponentialWeightFunction(), s::Int=8)
        @assert order == 2 "DumbserWENO currently only supports order=2."
        ws = DumbserWENOWorkspace(s)
        # res buffer is for the result of a single gradInterpolation! call
        new(order, zeros(5), weightFunction, s, ws)
    end
end


# --- 2. Refactored and Corrected DumbserWENO Functor for 2D ---
function (weno::DumbserWENO)(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    ws = weno.workspace
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    
    #if num_neighbors < 10; return 0.0; end # Heuristic check

    ensure_capacity!(ws, num_neighbors)
    dxVec = @view ws.dx_buffer[1:num_neighbors]
    dyVec = @view ws.dy_buffer[1:num_neighbors]
    dfVec = @view ws.df_buffer[1:num_neighbors]
    windowMatrix = @view ws.window_matrix[1:num_neighbors, :]

    fill!(windowMatrix, false)
    windowMatrix[:, 1] .= true

    for i in 1:num_neighbors
        nbIndex = neighbors[i]
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX / settings.interpRange
        dyVec[i] = deltaY / settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        stencil = getStencil(deltaX, deltaY, weno.s)
        windowMatrix[i, stencil + 2] = true
    end

    # --- Calculate Gradients for Each Stencil ---
    for stencil_idx in 1:(weno.s + 1)
        stencil_view = @view windowMatrix[:, stencil_idx]
        
        if count(stencil_view) < 5
            ws.gradients[:, stencil_idx] .= 1e10 
            continue
        end

        weno.weightFunction(wVec_stencil, (@view dxVec[stencil_view]), (@view dyVec[stencil_view]); param=settings.interpAlpha, normalisation=1.0)
        
        try
            gradInterpolation!((@view dxVec[stencil_view]), (@view dyVec[stencil_view]), wVec_stencil, (@view dfVec[stencil_view]), weno.res; order=weno.order)
            
            ws.gradients[1, stencil_idx] = weno.res[1] / settings.interpRange
            ws.gradients[2, stencil_idx] = weno.res[2] / settings.interpRange
            ws.gradients[3, stencil_idx] = weno.res[3] / (settings.interpRange^2)
            ws.gradients[4, stencil_idx] = weno.res[4] / (settings.interpRange^2)
            ws.gradients[5, stencil_idx] = weno.res[5] / (settings.interpRange^2)
        catch e
            if e isa SingularException
                ws.gradients[:, stencil_idx] .= 1e10
            else
                rethrow(e)
            end
        end
    end

    # --- Compute Non-Linear Weights (Corrected and Stabilized) ---
    r = 4
    eps = 1e-14
    
    for i in 1:(weno.s + 1)
        lambda = (i == 1) ? 1e5 : 1.0 # High weight for central stencil
        
        # A more robust smoothness indicator that is less sensitive to scaling
        smoothness = sum(ws.gradients[k, i]^2 for k in 1:5)
        
        ws.weights[i] = lambda / ((eps + smoothness)^r)
    end

    # Normalize weights
    sum_weights = sum(ws.weights)
    if sum_weights < 1e-14; return 0.0; end
    ws.weights ./= sum_weights
    
    if setCurvature 
        particleGrid.curvatures[particleIndex, 1] = dot(ws.weights, @view ws.gradients[3, :])
        particleGrid.curvatures[particleIndex, 2] = dot(ws.weights, @view ws.gradients[4, :])
    end

    vel = velocity(eq, 0.0)
    div_x = dot(ws.weights, @view ws.gradients[1, :])
    div_y = dot(ws.weights, @view ws.gradients[2, :])
    
    return div_x * vel[1] + div_y * vel[2]
end
