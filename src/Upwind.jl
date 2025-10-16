# ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  # Only relevant in 2D. In 1D, all algorithms are the same.
abstract type TiwariAlgorithm <: UpwindAlgorithm end  # Split domain in left and right for d/dx, and up and down for d/dy.
abstract type PraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type ClassicAlgorithm <: UpwindAlgorithm end  # Take all points 'behind' center point. 
abstract type RusanovAlgorithm <: UpwindAlgorithm end # This is no upwinding of course but easy implementation in this framework (numerical Flux given does not have to be upwind)

using InteractiveUtils

"""
    partition_workspace!(mask::AbstractBitVector, arrays::Vararg{AbstractVector})

Partitions multiple `arrays` in-place according to a boolean `mask`.
All elements where `mask` is true are moved to the front.
This is an O(N) operation with zero allocations.
"""
function partition_workspace!(mask, arrays::Vararg{AbstractVector})
    n = length(mask)
    if n == 0
        return
    end

    left, right = 1, n
    @inbounds while left <= right
        # Find the next `false` on the left side
        while left <= right && mask[left]
            left += 1
        end
        # Find the next `true` on the right side
        while left <= right && !mask[right]
            right -= 1
        end

        # If pointers haven't crossed, swap the out-of-place elements
        if left < right
            for arr in arrays
                arr[left], arr[right] = arr[right], arr[left]
            end
            # Also swap the mask itself to keep it consistent, though not strictly necessary
            mask[left], mask[right] = mask[right], mask[left] 
            
            left += 1
            right -= 1
        end
    end
end

# function populate_buffers!(dxVec, dyVec, dfVec, neighbors, xdist, ydist, fVec, vel, particleIndex)
#     count = 0
#     # Note: If your buffers aren't cleared, you'll need to manage count differently
#     @inbounds for i in 1:length(neighbors)
#         d1 = xdist[i]
#         d2 = ydist[i]
        
#         if d1 * vel[1] + d2 * vel[2] < 0
#             count += 1
#             nbIndex = neighbors[i]
            
#             dxVec[count] = d1
#             dyVec[count] = d2
#             dfVec[count] = fVec[nbIndex] - fVec[particleIndex]
#         end
#     end
#     return count # Return the number of items added
# end

function populate_buffers!(dxVec, dyVec, dfVec, neighbors, xdist, ydist, fVec, vel, particleIndex)
    
    # --- STEP 1: Vectorized Calculation (Branchless) ---
    # Perform the check for all neighbors at once using dot syntax.
    # This is extremely fast and uses SIMD.
    mask = (xdist .* vel[1] .+ ydist .* vel[2]) .< 0
    
    # --- STEP 2: Find Indices That Passed the Test ---
    # `findall` returns the indices where the mask is `true`.
    passing_indices = findall(mask)
    
    # --- STEP 3: Gather Data Using the Filtered Indices ---
    # This loop is much shorter and contains no 'if' statement.
    count = length(passing_indices)
    @inbounds for i in 1:count
        # The index into the original neighbor arrays
        original_idx = passing_indices[i]
        
        # Get the neighbor's global index
        nbIndex = neighbors[original_idx]
        
        # Populate the final buffers
        dxVec[i] = xdist[original_idx]
        dyVec[i] = ydist[original_idx]
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end
    
    return count
end

# Define a workspace to hold temporary arrays for Upwind calculations
struct UpwindWorkspace
    dxVec::Vector{Float64}
    dyVec::Vector{Float64}
    dfVec::Vector{Float64}
    fVec::Vector{Float64}
    wVec::Vector{Float64}
    # For Tiwari algorithm
    xWindow::BitVector
    yWindow::BitVector
    # For PraveenAlgorithm
    coeff_x_Vec::Vector{Float64}
    coeff_y_Vec::Vector{Float64}
    aij_x_Vec::Vector{Float64}
    aij_y_Vec::Vector{Float64}
    nxVec::Vector{Float64}
    nyVec::Vector{Float64}
    # Add a buffer for neighbor values
    ujVec::Vector{Float64}
    nb_buffer::Vector{Int}

    function UpwindWorkspace(max_neighbors::Int=100) # Preallocate with a reasonable capacity
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            falses(max_neighbors),
            falses(max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Int}(undef, max_neighbors),
        )
    end
end

# Helper to ensure workspace vectors are large enough
function ensure_capacity!(ws::UpwindWorkspace, n::Int)
    if length(ws.dxVec) < n
        N = n + n ÷ 4
        resize!.((ws.dxVec, ws.dyVec, ws.dfVec, ws.fVec, ws.wVec, ws.xWindow, ws.yWindow, ws.coeff_x_Vec, ws.coeff_y_Vec, ws.aij_x_Vec, ws.aij_y_Vec, ws.nxVec, ws.nyVec, ws.ujVec, ws.nb_buffer), N)
    end
end


struct UpwindGradient{D, WS <: UpwindWorkspace, I <: Interpolator, Algorithm <: UpwindAlgorithm} <: GradientInterpolator
    order::Int
    weightFunction::MLSWeightFunction
    numericalFlux::NumericalFluxFunction
    workspace::UpwindWorkspace
    interpolator::I

    """
        UpwindGradient(order::Int64 = 1; algType::String = "")

    Constructor for Upwind Object. algType only has impact in 2D upwinding.
    """
    function UpwindGradient(order, dimension; numericalFlux::NumericalFluxFunction=UpwindFlux(), algType::String="Classic", weightFunction::MLSWeightFunction=exponentialWeightFunction())
        @assert order >= 1 "Order must be larger or equal to one."
        @assert algType in ["Classic", "Tiwari", "Praveen", "NonLinearPraveen"]
        local alg_type
        if algType == "Classic"
            alg_type = ClassicAlgorithm
        elseif algType == "Praveen"
            @assert order == 1
            alg_type = PraveenAlgorithm
        elseif algType == "NonLinearPraveen"
            @assert order == 1
            alg_type = NonLinearPraveenAlgorithm
        elseif algType == "Tiwari"
            alg_type = TiwariAlgorithm
        end
        ws = UpwindWorkspace()
        interpolator = Interpolator{dimension, order, 1}()
        WS = typeof(ws)
        I = typeof(interpolator)

        new{dimension, WS, I, alg_type}(order, weightFunction, numericalFlux, ws, interpolator)
    end
end

# ... (keep all other content in Interpolations.jl) ...

#==============================================================================
  UPWIND GRADIENT (Optimized for SoA Grids)
==============================================================================#

# --- REFACTORED 1D Upwind Functor ---
function (upwind::UpwindGradient{1,WS,I,A})(
    particleGrid::ParticleGrid1D,
    particleIndex::Integer,
    fVec::AbstractVector{<:Real},
    eq::ScalarHyperbolicPDE,
    settings::SimSetting;
    setCurvature::Bool=true
)::Real where {WS <: UpwindWorkspace, I <: Interpolator, A <: UpwindAlgorithm}
    
    neighbors = particleGrid.neighbor_indices[particleIndex]
    num_neighbors = length(neighbors)
    ws = upwind.workspace
    interp = upwind.interpolator
    ensure_capacity!(ws, num_neighbors)
    ensure_capacity!(interp, num_neighbors)

    # Use zero-cost views into the workspace buffers
    dxVec = @view ws.dxVec[1:num_neighbors]
    dfVec = @view ws.dfVec[1:num_neighbors]
    wVec = @view ws.wVec[1:num_neighbors]

    for (i, nbIndex) in enumerate(neighbors)
        deltaPos = getDistance(particleGrid, particleIndex, nbIndex)
        fm, fp = sortFlux(fVec[particleIndex], fVec[nbIndex], deltaPos)
        
        dxVec[i] = deltaPos / settings.interpRange
        dfVec[i] = upwind.numericalFlux(fm, fp, eq) - flux(eq, fVec[particleIndex])
    end
    
    upwind.weightFunction(wVec, dxVec; param=settings.interpAlpha, normalisation=1.0)
    res1 = interp(dxVec, wVec, dfVec)

    if setCurvature
        particleGrid.curvatures[particleIndex] = 0.0
    end
    
    return 2 * res1 / settings.interpRange
end


function (upwind::UpwindGradient{2,WS,I,ClassicAlgorithm})(
    pg::ParticleGrid2D{S},
    particleIndex::Int,
    fVec_inp::AbstractVector{<:Real},
    eq::LinearAdvection{2},
    settings::SimSetting;
    setCurvature::Bool=true
)::Real where {S,WS <: UpwindWorkspace, I <: Interpolator}
    
    vel = eq.vel
    ws = upwind.workspace
    interp = upwind.interpolator
    
    # --- 1. Get neighbor data slices from the main grid ---
    start_idx = pg.neighbor_pointers[particleIndex]
    num_neighbors = pg.num_neighbors[particleIndex]
    if num_neighbors == 0; return 0.0; end
    ensure_capacity!(ws, num_neighbors)
    neighbor_slice = start_idx:(start_idx + num_neighbors - 1)
    
    # Create views into the large, persistent particle grid data. No copies here.
    all_neighbors = @view pg.neighbor_indices[neighbor_slice]
    all_dx = @view pg.neighbor_xdistance[neighbor_slice]
    all_dy = @view pg.neighbor_ydistance[neighbor_slice]

    # --- 2. The Filter & Compact Loop (Zero Allocations) ---
    # This loop reads from the grid and writes only the "upwind" data
    # sequentially into the start of the workspace buffers.
    count = 0
    f_particle = fVec_inp[particleIndex]
    
    @inbounds for i in 1:num_neighbors
        # Condition check is simple and cheap
        tmp = all_dx[i] * vel[1]
        if tmp + all_dy[i] * vel[2] < 0
            count += 1
            ws.dxVec[count] = all_dx[i]
            ws.dyVec[count] = all_dy[i]
            
            nbIndex = all_neighbors[i]
            ws.dfVec[count] = fVec_inp[nbIndex] - f_particle
        end
    end

    num_upwind = count
    if num_upwind < upwind.order; return 0.0; end
    
    ensure_capacity!(interp, num_upwind)

    # --- 3. Process the Compacted Data ---
    # Create views into the small, dense, cache-friendly data we just collected.
    dxVec_upwind = @view ws.dxVec[1:num_upwind]
    dyVec_upwind = @view ws.dyVec[1:num_upwind]
    dfVec_upwind = @view ws.dfVec[1:num_upwind]
    wVec_upwind  = @view ws.wVec[1:num_upwind]

    upwind.weightFunction(wVec_upwind, dxVec_upwind, dyVec_upwind; param=settings.interpAlpha, normalisation=1.0)
    
    local res1, res2
    if upwind.order == 1
        res1, res2 = interp(dxVec_upwind, dyVec_upwind, wVec_upwind, dfVec_upwind)
    elseif upwind.order == 2
        res1, res2, res3, res4 = interp(dxVec_upwind, dyVec_upwind, wVec_upwind, dfVec_upwind)
        if setCurvature
            pg.curvatures[particleIndex, 1] = res3 / (settings.interpRange^2)
            pg.curvatures[particleIndex, 2] = res4 / (settings.interpRange^2)
        end
    end
    
    return (vel[1] * res1 + vel[2] * res2) / settings.interpRange
end

# # --- REFACTORED 2D Upwind Functor (Classic Algorithm) ---
# function (upwind::UpwindGradient{2,WS,I,ClassicAlgorithm})(
#     pg::ParticleGrid2D{S},
#     particleIndex::Int,
#     fVec_inp::AbstractVector{<:Real},
#     eq::LinearAdvection{2},
#     settings::SimSetting;
#     setCurvature::Bool=true
# )::Real where {S,WS <: UpwindWorkspace, I <: Interpolator}
    
#     vel = eq.vel
#     ws = upwind.workspace
#     interp = upwind.interpolator
#     # --- NEW, FAST WAY ---
#     start_idx = pg.neighbor_pointers[particleIndex]
#     end_idx   = pg.neighbor_pointers[particleIndex + 1] - 1
#     neighbor_slice = start_idx:end_idx
#     neighbors = @view pg.neighbor_indices[neighbor_slice]
#     num_neighbors = pg.num_neighbors[particleIndex]

#     ensure_capacity!(ws, num_neighbors)

#     dxVec = @view ws.dxVec[1:num_neighbors]
#     dyVec = ws.dyVec[1:num_neighbors]
#     dfVec = ws.dfVec[1:num_neighbors]
#     fVec = ws.fVec[1:num_neighbors]
#     wVec = ws.wVec[1:num_neighbors]
#     upwinding = ws.xWindow[1:num_neighbors]
#     buffer = ws.ujVec[1:num_neighbors]

#     fVal = fVec_inp[particleIndex]

#     dxVec .= pg.neighbor_xdistance[neighbor_slice]
#     dyVec .= pg.neighbor_ydistance[neighbor_slice]
#     fVec .= fVec_inp[neighbors]
#     buffer .= dxVec * vel[1]
#     buffer .+= dyVec * vel[2]
#     dfVec .= fVec .+ fVal

#     upwinding .= 0
#     upwinding = buffer .< 0.
#     #@code_warntype populate_buffers!(dxVec, dyVec, dfVec, neighbors, xdist, ydist, fVec, vel, particleIndex)
#     #error("TEST")
# # --- THE FIX: The "Gather" Loop ---
#     # In a single pass, identify upwind neighbors and gather ALL their data
#     # into the small, contiguous workspace buffers.
#     # count = 0
#     # @inbounds for i in 1:num_neighbors
#     #     if dxVec[i] * vel[1] + dyVec[i] * vel[2] < 0
#     #         count += 1
#     #         nbIndex = neighbors[i]
#     #         dfVec[count] = fVec[nbIndex] - fVec[particleIndex]
#     #     end
#     # end

#     num_upwind = sum(upwinding)
#     if num_upwind < upwind.order; return 0.0; end
    
#     ensure_capacity!(interp, num_upwind)

#     # --- The "Process" Step ---
#     # All subsequent operations are on small, dense, cache-friendly buffers.
#     # All memory access from here is sequential and extremely fast.
#     dxVec = @view dxVec[upwinding]
#     dyVec = @view dyVec[upwinding]
#     dfVec = @view dfVec[upwinding]
#     wVec  = @view wVec[upwinding]

#     upwind.weightFunction(wVec, dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
#     local res1, res2
#     if upwind.order == 1
#         res1, res2 = interp(dxVec, dyVec, wVec, dfVec)
#     elseif upwind.order == 2
#         res1, res2, res3, res4 = interp(dxVec, dyVec, wVec, dfVec)

#         if setCurvature
#             pg.curvatures[particleIndex, 1] = res3 / (settings.interpRange^2)
#             pg.curvatures[particleIndex, 2] = res4 / (settings.interpRange^2)
#         end
#     end
    
#     return (vel[1] * res1  + vel[2] * res2) / settings.interpRange
# end

# function (upwind::UpwindGradient{2,WS,I,ClassicAlgorithm})(
#     pg::ParticleGrid2D{S},
#     particleIndex::Int,
#     fVec_inp::AbstractVector{<:Real},
#     eq::LinearAdvection{2},
#     settings::SimSetting;
#     setCurvature::Bool=true
# )::Real where {S,WS <: UpwindWorkspace, I <: Interpolator}
    
#     vel = eq.vel
#     ws = upwind.workspace
#     interp = upwind.interpolator

#     # --- 1. Get views into neighbor data and workspace buffers ---
#     start_idx = pg.neighbor_pointers[particleIndex]
#     num_neighbors = pg.num_neighbors[particleIndex]
#     if num_neighbors == 0; return 0.0; end
#     neighbor_slice = start_idx:(start_idx + num_neighbors - 1)

#     ensure_capacity!(ws, num_neighbors)

#     # These are all views into pre-allocated memory
#     dxVec = @view ws.dxVec[1:num_neighbors]
#     dyVec = @view ws.dyVec[1:num_neighbors]
#     dfVec = @view ws.dfVec[1:num_neighbors]
#     wVec = @view ws.wVec[1:num_neighbors]
#     buffer = ws.ujVec[1:num_neighbors]
#     upwinding_mask = @view ws.xWindow[1:num_neighbors] # Use the BitVector from workspace

#     # --- 2. Populate workspace and create the boolean mask (Vectorized & Allocation-Free) ---
#     dxVec .= @view pg.neighbor_xdistance[neighbor_slice]
#     dyVec .= @view pg.neighbor_ydistance[neighbor_slice]

#     buffer .= dxVec * vel[1]
#     buffer .+= dyVec * vel[2]
    
#     # Use a temporary view for neighbors to populate dfVec
#     neighbors_view = @view pg.neighbor_indices[neighbor_slice]
#     f_particle = fVec_inp[particleIndex]
#     dfVec .= (@view fVec_inp[neighbors_view]) .- f_particle

#     # Calculate the upwind condition and store it in the pre-allocated BitVector
#     upwinding_mask .= buffer .< 0

#     num_upwind = sum(upwinding_mask)
#     if num_upwind < upwind.order; return 0.0; end

#     # --- 3. Partition all workspace arrays based on the mask (The Efficient Sort) ---
#     partition_workspace!(upwinding_mask, dxVec, dyVec, dfVec) # wVec is not needed yet

#     ensure_capacity!(interp, num_upwind)

#     # --- 4. Process the compacted data ---
#     # The data we need is now guaranteed to be in the first `num_upwind` slots.
#     # Create cheap views into this dense, cache-friendly data.
#     dxVec_upwind = @view dxVec[1:num_upwind]
#     dyVec_upwind = @view dyVec[1:num_upwind]
#     dfVec_upwind = @view dfVec[1:num_upwind]
#     wVec_upwind  = @view wVec[1:num_upwind] # wVec is only used here

#     upwind.weightFunction(wVec_upwind, dxVec_upwind, dyVec_upwind; param=settings.interpAlpha, normalisation=1.0)
    
#     local res1, res2
#     if upwind.order == 1
#         res1, res2 = interp(dxVec_upwind, dyVec_upwind, wVec_upwind, dfVec_upwind)
#     elseif upwind.order == 2
#         res1, res2, res3, res4 = interp(dxVec_upwind, dyVec_upwind, wVec_upwind, dfVec_upwind)
#         if setCurvature
#             pg.curvatures[particleIndex, 1] = res3 / (settings.interpRange^2)
#             pg.curvatures[particleIndex, 2] = res4 / (settings.interpRange^2)
#         end
#     end
    
#     return (vel[1] * res1 + vel[2] * res2) / settings.interpRange
# end

function (upwind::UpwindGradient{2,WS ,I ,TiwariAlgorithm})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector{<:Real}, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real where {WS <: UpwindWorkspace, I <: Interpolator}
    
    vel = velocity(eq, 0.0)
    ws = upwind.workspace
    interp = upwind.interpolator
    neighbor_indices = particleGrid.neighbor_indices[particleIndex]
    num_neighbors = length(neighbor_indices)

    if num_neighbors < upwind.order
        if setCurvature; particleGrid.curvatures[particleIndex, :] .= 0.0; end
        return 0.0
    end

    ensure_capacity!(ws, num_neighbors)

    # --- 1. Data Collection (Single Pass) ---
    # Get views into the main workspace buffers
    dxVec = @view ws.dxVec[1:num_neighbors]
    dyVec = @view ws.dyVec[1:num_neighbors]
    dfVec = @view ws.dfVec[1:num_neighbors]
    xWindow = @view ws.xWindow[1:num_neighbors] # Reuse left_window as xWindow
    yWindow = @view ws.yWindow[1:num_neighbors]  # Reuse top_window as yWindow

    for (i, nbIndex) in enumerate(neighbor_indices)
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX / settings.interpRange
        dyVec[i] = deltaY / settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        xWindow[i] = (vel[1] * deltaX <= 0.0) # Simplified upwind condition
        yWindow[i] = (vel[2] * deltaY <= 0.0) # Simplified upwind condition
    end

    # --- 2. X-Derivative Calculation ---
    stencil_size_x = 0
    for i in 1:num_neighbors
        if xWindow[i]
            stencil_size_x += 1
            ws.dxVec[stencil_size_x] = dxVec[i]
            ws.dyVec[stencil_size_x] = dyVec[i]
            ws.dfVec[stencil_size_x] = dfVec[i]
        end
    end

    ddx = 0.0
    if stencil_size_x >= upwind.order
        # Create views of the filtered data in the scratch buffers
        dx_stencil_x = @view ws.dxVec[1:stencil_size_x]
        dy_stencil_x = @view ws.dyVec[1:stencil_size_x]
        df_stencil_x = @view ws.dfVec[1:stencil_size_x]
        w_stencil_x  = @view ws.wVec[1:stencil_size_x]

        ensure_capacity!(interp, stencil_size_x)
        upwind.weightFunction(w_stencil_x, dx_stencil_x, dy_stencil_x; param=settings.interpAlpha, normalisation=1.0)
        res_x = interp(dx_stencil_x, dy_stencil_x, w_stencil_x, df_stencil_x)
        
        ddx = res_x[1] / settings.interpRange
        if setCurvature && upwind.order == 2
            particleGrid.curvatures[particleIndex, 1] = res_x[3] / (settings.interpRange^2)
        end
    end
        
    # --- 3. Y-Derivative Calculation ---
    stencil_size_y = 0
    for i in 1:num_neighbors
        if yWindow[i]
            stencil_size_y += 1
            # REUSE the same scratch buffers
            ws.dxVec[stencil_size_y] = dxVec[i]
            ws.dyVec[stencil_size_y] = dyVec[i]
            ws.dfVec[stencil_size_y] = dfVec[i]
        end
    end
    
    ddy = 0.0
    if stencil_size_y >= upwind.order
        # Create views of the filtered data in the scratch buffers
        dx_stencil_y = @view ws.dxVec[1:stencil_size_y]
        dy_stencil_y = @view ws.dyVec[1:stencil_size_y]
        df_stencil_y = @view ws.dfVec[1:stencil_size_y]
        w_stencil_y  = @view ws.wVec[1:stencil_size_y]
        
        ensure_capacity!(interp, stencil_size_y)
        upwind.weightFunction(w_stencil_y, dx_stencil_y, dy_stencil_y; param=settings.interpAlpha, normalisation=1.0)
        res_y = interp(dx_stencil_y, dy_stencil_y, w_stencil_y, df_stencil_y)

        ddy = res_y[2] / settings.interpRange
        if setCurvature && upwind.order == 2
            particleGrid.curvatures[particleIndex, 2] = res_y[4] / (settings.interpRange^2)
        end
    end

    # --- 4. Final Result ---
    return ddx * vel[1] + ddy * vel[2]
end


# ... (existing content of Interpolations.jl, including the UpwindWorkspace) ...

#==============================================================================
  Additional 2D Upwind Functors (Optimized for SoA Grids & Workspace)
==============================================================================#

function (upwind::UpwindGradient{2, WS, I, PraveenAlgorithm})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector{<:Real}, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real where {WS <: UpwindWorkspace, I <: Interpolator}
    
    vel = velocity(eq, 0.0)
    neighbor_indices = particleGrid.neighbor_indices[particleIndex]
    num_neighbors = length(neighbor_indices)
    
    if num_neighbors == 0; return 0.0; end

    if setCurvature
        particleGrid.curvatures[particleIndex, :] .= 0.0
    end
    
    # --- 1. Data Collection (Single Pass) ---
    ws = upwind.workspace
    ensure_capacity!(ws, num_neighbors)
    ui = fVec[particleIndex]
    for (i, nbIndex) in enumerate(neighbor_indices)
        ws.dxVec[i], ws.dyVec[i] = getDistance(particleGrid, particleIndex, nbIndex)
        ws.dfVec[i] = fVec[nbIndex] - ui
    end

    # Create views for the exact number of neighbors
    dxVec = @view ws.dxVec[1:num_neighbors]
    dyVec = @view ws.dyVec[1:num_neighbors]
    dfVec = @view ws.dfVec[1:num_neighbors]
    wVec  = @view ws.wVec[1:num_neighbors]
    
    # --- 2. Least-Squares System Setup (Vectorized) ---
    upwind.weightFunction(wVec, dxVec, dyVec; param=settings.interpAlpha, normalisation=settings.interpRange)

    A11 = sum(wVec[i] * dxVec[i]^2 for i in 1:num_neighbors)
    A22 = sum(wVec[i] * dyVec[i]^2 for i in 1:num_neighbors)
    A12 = sum(wVec[i] * dxVec[i] * dyVec[i] for i in 1:num_neighbors)
    D = A11 * A22 - A12^2
    
    if abs(D) < 1e-14; return 0.0; end
    
    # --- 3. Divergence Calculation (Vectorized) ---
    # Get views for all necessary buffers
    coeff_x_Vec = @view ws.coeff_x_Vec[1:num_neighbors]
    coeff_y_Vec = @view ws.coeff_y_Vec[1:num_neighbors]
    nxVec       = @view ws.nxVec[1:num_neighbors]
    nyVec       = @view ws.nyVec[1:num_neighbors]
    
    # Solve for coefficients (vectorized)
    coeff_x_Vec .= (wVec .* (A22 .* dxVec .- A12 .* dyVec)) ./ D
    coeff_y_Vec .= (wVec .* (A11 .* dyVec .- A12 .* dxVec)) ./ D

    # Rotational vectors (vectorized)
    angles = atan.(dyVec, dxVec)
    nxVec .= cos.(angles)
    nyVec .= sin.(angles)
    # sxVec is -nyVec, syVec is nxVec

    # Compute adapted coefficients and positivity terms (vectorized)
    # Reuse aij buffers for alfaBar and betaBar
    alfaBarVec = @view ws.aij_x_Vec[1:num_neighbors]
    betaBarVec = @view ws.aij_y_Vec[1:num_neighbors]
    
    alfaBarVec .= nxVec .* coeff_x_Vec .+ nyVec .* coeff_y_Vec
    betaBarVec .= (-nyVec) .* coeff_x_Vec .+ nxVec .* coeff_y_Vec
    
    # Reuse coeff buffers for temporary dot products with velocity
    dot_vel_n = coeff_x_Vec # Rename for clarity
    dot_vel_s = coeff_y_Vec
    
    dot_vel_n .= vel[1] .* nxVec .+ vel[2] .* nyVec
    dot_vel_s .= vel[1] .* (-nyVec) .+ vel[2] .* nxVec

    # Calculate bracket terms using min/max for conditional logic
    bracketMinusVec  = min.(dot_vel_n, 0.0)
    bracketMinus2Vec = min.(betaBarVec .* dot_vel_s, 0.0)
    
    # Calculate final coefficient vector `cij`
    # Reuse alfaBarVec buffer to store the final `cij` values
    cijVec = alfaBarVec
    cijVec .*= bracketMinusVec
    cijVec .+= bracketMinus2Vec
    
    # Final divergence is a single dot product
    div = 2 * dot(cijVec, dfVec)
    
    return div
end


function (upwind::UpwindGradient{2,WS,I,NonLinearPraveenAlgorithm})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector{<:Real}, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real where {WS <: UpwindWorkspace, I <: Interpolator}
    
    vel = velocity(eq, 0.0)
    neighbor_indices = particleGrid.neighbor_indices[particleIndex]
    num_neighbors = length(neighbor_indices)
    
    if num_neighbors == 0; return 0.0; end

    if setCurvature
        particleGrid.curvatures[particleIndex, :] .= 0.0
    end
    
    # --- 1. Data Collection ---
    ws = upwind.workspace
    ensure_capacity!(ws, num_neighbors)
    ui = fVec[particleIndex]
    for (i, nbIndex) in enumerate(neighbor_indices)
        ws.dxVec[i], ws.dyVec[i] = getDistance(particleGrid, particleIndex, nbIndex)
        ws.ujVec[i] = fVec[nbIndex] # Store uj directly
    end

    # Create views
    dxVec = @view ws.dxVec[1:num_neighbors]
    dyVec = @view ws.dyVec[1:num_neighbors]
    ujVec = @view ws.ujVec[1:num_neighbors]
    wVec  = @view ws.wVec[1:num_neighbors]
    
    # --- 2. Least-Squares System ---
    upwind.weightFunction(wVec, dxVec, dyVec; param=settings.interpAlpha, normalisation=settings.interpRange)

    A11 = sum(wVec[i] * dxVec[i]^2 for i in 1:num_neighbors)
    A22 = sum(wVec[i] * dyVec[i]^2 for i in 1:num_neighbors)
    A12 = sum(wVec[i] * dxVec[i] * dyVec[i] for i in 1:num_neighbors)
    D = A11 * A22 - A12^2
    
    if abs(D) < 1e-14; return 0.0; end
    
    # --- 3. Divergence Calculation ---
    # Get views for buffers
    coeff_x_Vec = @view ws.coeff_x_Vec[1:num_neighbors]
    coeff_y_Vec = @view ws.coeff_y_Vec[1:num_neighbors]
    nxVec       = @view ws.nxVec[1:num_neighbors]
    nyVec       = @view ws.nyVec[1:num_neighbors]

    # --- THE FIX: Calculate flux components separately ---
    # For LinearAdvection, flux(u) = velocity * u. We can vectorize this directly.
    # Reuse aij buffers for the neighbor fluxes.
    fj_x_Vec = @view ws.aij_x_Vec[1:num_neighbors]
    fj_y_Vec = @view ws.aij_y_Vec[1:num_neighbors]

    fj_x_Vec .= vel[1] .* ujVec
    fj_y_Vec .= vel[2] .* ujVec
    
    # --- The rest of the vectorized logic follows ---

    # We need (uj - ui), which is (ujVec .- ui). Store it in dfVec.
    dfVec = @view ws.dfVec[1:num_neighbors]
    dfVec .= ujVec .- ui

    # The rest of the function remains the same, as it was already correct.
    coeff_x_Vec .= (wVec .* (A22 .* dxVec .- A12 .* dyVec)) ./ D
    coeff_y_Vec .= (wVec .* (A11 .* dyVec .- A12 .* dxVec)) ./ D
    
    angles = atan.(dyVec, dxVec)
    nxVec .= cos.(angles)
    nyVec .= sin.(angles)
    
    alfaBarVec = @view ws.aij_x_Vec[1:num_neighbors] # Reuse buffer
    betaBarVec = @view ws.aij_y_Vec[1:num_neighbors] # Reuse buffer
    
    alfaBarVec .= nxVec .* coeff_x_Vec .+ nyVec .* coeff_y_Vec
    betaBarVec .= (-nyVec) .* coeff_x_Vec .+ nxVec .* coeff_y_Vec
    
    dot_vel_n = coeff_x_Vec # Reuse buffer
    dot_vel_s = coeff_y_Vec # Reuse buffer
    
    dot_vel_n .= vel[1] .* nxVec .+ vel[2] .* nyVec
    dot_vel_s .= vel[1] .* (-nyVec) .+ vel[2] .* nxVec

    bracketMinusVec  = min.(dot_vel_n, 0.0)
    bracketMinus2Vec = min.(betaBarVec .* dot_vel_s, 0.0)
    
    cijVec = alfaBarVec # Reuse buffer
    cijVec .*= bracketMinusVec
    cijVec .+= bracketMinus2Vec
    
    div = 2 * dot(cijVec, dfVec)
    
    return div
end

