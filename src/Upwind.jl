# ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  # Only relevant in 2D. In 1D, all algorithms are the same.
abstract type TiwariAlgorithm <: UpwindAlgorithm end  # Split domain in left and right for d/dx, and up and down for d/dy.
abstract type PraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type ClassicAlgorithm <: UpwindAlgorithm end  # Take all points 'behind' center point. 
abstract type RusanovAlgorithm <: UpwindAlgorithm end # This is no upwinding of course but easy implementation in this framework (numerical Flux given does not have to be upwind)

abstract type UpwindWorkspace end

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
    for i in 1:count
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

# # Define a workspace to hold temporary arrays for Upwind calculations
# struct UpwindWorkspace
#     dxVec::Vector{Float64}
#     dyVec::Vector{Float64}
#     dfVec::Vector{Float64}
#     fVec::Vector{Float64}
#     wVec::Vector{Float64}
#     # For Tiwari algorithm
#     xWindow::BitVector
#     yWindow::BitVector
#     # For PraveenAlgorithm
#     coeff_x_Vec::Vector{Float64}
#     coeff_y_Vec::Vector{Float64}
#     aij_x_Vec::Vector{Float64}
#     aij_y_Vec::Vector{Float64}
#     nxVec::Vector{Float64}
#     nyVec::Vector{Float64}
#     # Add a buffer for neighbor values
#     ujVec::Vector{Float64}
#     nb_buffer::Vector{Int}

#     function UpwindWorkspace(max_neighbors::Int=100) # Preallocate with a reasonable capacity
#         new(
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             falses(max_neighbors),
#             falses(max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Float64}(undef, max_neighbors),
#             Vector{Int}(undef, max_neighbors),
#         )
#     end
# end


"""
A thread-local workspace for the Upwind TiwariAlgorithm.
Holds temporary buffers for all neighbors and for the filtered stencils.
"""
struct UpwindWorkspaceTA <: UpwindWorkspace
    # Buffers to hold ALL neighbor data initially (size num_neighbors)
    dxVec::Vector{Float64}
    dyVec::Vector{Float64}
    dfVec::Vector{Float64}
    wVec::Vector{Float64}  # For calculated weights

    # BitVectors to mark upwind neighbors for each direction
    xWindow::BitVector
    yWindow::BitVector

    function UpwindWorkspaceTA(max_neighbors::Int=100) # Preallocate
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            falses(max_neighbors),
            falses(max_neighbors)
        )
    end
end

"""
A minimal, thread-local workspace for the Upwind ClassicAlgorithm.
It holds temporary buffers for the filtered "upwind" neighbors.
"""
struct UpwindWorkspaceCA <: UpwindWorkspace
    dxVec::Vector{Float64} # Filtered dx (upwind)
    dyVec::Vector{Float64} # Filtered dy (upwind)
    dfVec::Vector{Float64} # Filtered df (upwind)
    wVec::Vector{Float64}  # Filtered w (upwind)

    function UpwindWorkspaceCA(max_neighbors::Int=100) # Preallocate
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors)
        )
    end
end

"""
A minimal, thread-local workspace for the Upwind PraveenAlgorithm.
Stores only essential coefficients.
"""
struct UpwindWorkspacePA <: UpwindWorkspace
    # Buffers to hold calculated coefficients (size num_neighbors)
    coeff_x_Vec::Vector{Float64}
    coeff_y_Vec::Vector{Float64}
    cijVec::Vector{Float64}     # Final coefficient

    function UpwindWorkspacePA(max_neighbors::Int=100) # Preallocate
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors)
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
"""
Ensures all buffers in the minimal UpwindWorkspaceCA are large enough.
"""
function ensure_capacity!(ws::UpwindWorkspaceCA, n::Int)
    _ensure_capacity!(ws.dxVec, n)
    _ensure_capacity!(ws.dyVec, n)
    _ensure_capacity!(ws.dfVec, n)
    _ensure_capacity!(ws.wVec, n)
end

# --- Modify the UpwindGradient Constructor ---
function UpwindGradient(order, dimension; numericalFlux::NumericalFluxFunction=UpwindFlux(), algType::String="Classic", weightFunction::MLSWeightFunction=exponentialWeightFunction())
    # ... (assertions) ...
    
    local alg_type
    local WS_eltype::Type 
    
    if algType == "Classic"
        alg_type = ClassicAlgorithm
        WS_eltype = UpwindWorkspaceCA 
    elseif algType == "Tiwari"
         alg_type = TiwariAlgorithm
         WS_eltype = UpwindWorkspaceTA 
    elseif algType == "Praveen"
        alg_type = PraveenAlgorithm # <-- NEW
        WS_eltype = UpwindWorkspacePA # <-- NEW
        @assert order == 1
    # elseif algType == "NonLinearPraveen" 
    #     alg_type = NonLinearPraveenAlgorithm
    #     WS_eltype = UpwindWorkspace # Or create UpwindWorkspaceNLPA
    #     @assert order == 1
    else
         error("Algorithm type $algType not fully configured for workspace selection.")
    end

    n_threads = Threads.nthreads()
    workspaces = [WS_eltype(100) for _ in 1:n_threads] 

    interpolator = Interpolator{dimension, order, 1}()
    I = typeof(interpolator)

    new{dimension, WS_eltype, I, alg_type}(order, weightFunction, numericalFlux, workspaces, interpolator)
end
"""
Ensures all thread-local workspaces in the UpwindGradient object are
correctly sized and re-initializes them if the thread count changed.
"""
function ensure_capacity!(g::UpwindGradient{<:Any, UpwindWorkspaceCA, <:Any, <:Any}, max_neighbors::Int)
    n_threads = Threads.nthreads()
    
    # Re-create workspaces if thread count changed
    if length(g.workspaces) != n_threads
        empty!(g.workspaces)
        for _ in 1:n_threads
            push!(g.workspaces, UpwindWorkspaceCA(max_neighbors))
        end
    end
    
    # Ensure all workspaces have enough capacity
    for ws in g.workspaces
        ensure_capacity!(ws, max_neighbors)
    end
end

# Uses the simple _ensure_capacity! helper from your code
"""
Ensures all buffers in the UpwindWorkspaceTA are large enough.
"""
function ensure_capacity!(ws::UpwindWorkspaceTA, n::Int)
    _ensure_capacity!(ws.dxVec, n)
    _ensure_capacity!(ws.dyVec, n)
    _ensure_capacity!(ws.dfVec, n)
    _ensure_capacity!(ws.wVec, n)
    _ensure_capacity!(ws.xWindow, n)
    _ensure_capacity!(ws.yWindow, n)
end

# Uses the simple _ensure_capacity! helper
"""
Ensures all buffers in the minimal UpwindWorkspacePA are large enough.
"""
function ensure_capacity!(ws::UpwindWorkspacePA, n::Int)
    _ensure_capacity!(ws.coeff_x_Vec, n)
    _ensure_capacity!(ws.coeff_y_Vec, n)
    _ensure_capacity!(ws.cijVec, n)
end

# The _init_buffers_internal! and initGIBuffers! functions 
# will adapt correctly based on the dispatch for ensure_capacity!

# Add dispatch for the Tiwari workspace vector to the existing helper
"""
Helper function to resize a vector of workspaces.
Handles thread-count changes and resizes all individual workspaces.
"""
function _init_buffers_internal!(workspaces::Vector{WS}, max_neighbors::Int) where WS <: UpwindWorkspace # Adjust constraint if needed
    n_threads = Threads.nthreads()
    
    if length(workspaces) != n_threads
        empty!(workspaces)
        for _ in 1:n_threads
            push!(workspaces, WS(max_neighbors)) 
        end
    end
    
    for ws in workspaces
        ensure_capacity!(ws, max_neighbors) # Calls correct overload based on WS
    end
end



# The main initGIBuffers! function remains unchanged, as it uses the helper above.
"""
Buffer initialization hook for UpwindGradient. Finds the max neighbors
[cite_start]from the grid and resizes all thread-local buffers. [cite: 30, 35]
"""
function initGIBuffers!(g::UpwindGradient, pg::ParticleGrid)
    max_nb = 0
    if !isempty(pg.num_neighbors)
        max_nb = maximum(pg.num_neighbors)
    end
    _init_buffers_internal!(g.workspaces, max_nb)
end


function initGI!(g::UpwindGradient, kwargs...)
    return
end



function initTimeStep(pg::ParticleGrid, weightFunc::MLSWeightFunction)
    updateNeighbors!(pg, weightFunc)
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

"""
Helper function to resize a vector of workspaces.
Handles thread-count changes and resizes all individual workspaces.
"""
function _init_buffers_internal!(workspaces::Vector{WS}, max_neighbors::Int) where WS
    n_threads = Threads.nthreads()
    # Re-create workspaces if thread count changed
    if length(workspaces) != n_threads
        empty!(workspaces)
        for _ in 1:n_threads
            push!(workspaces, WS(max_neighbors)) # Create new ones
        end
    end
    
    # Ensure all workspaces have enough capacity
    for ws in workspaces
        ensure_capacity!(ws, max_neighbors) # Calls the correct overload
    end
end

"""
Buffer initialization hook for UpwindGradient. Finds the max neighbors
from the grid and resizes all thread-local buffers.
"""
function initGIBuffers!(g::UpwindGradient, pg::ParticleGrid)
    max_nb = 0
    if !isempty(pg.num_neighbors)
        # Find the maximum number of neighbors any particle has
        max_nb = maximum(pg.num_neighbors)
    end
    
    # Dispatch to the internal helper
    _init_buffers_internal!(g.workspaces, max_nb)
end

"""
Functor for UpwindGradient (ClassicAlgorithm) with the 'fused' signature.
Calculates the upwind gradient for a single particle `i`.
Uses pg.range_factor for scaling.
"""
function (upwind::UpwindGradient{2, <:UpwindWorkspaceCA, <:Any, ClassicAlgorithm})(
    eq::ScalarHyperbolicPDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid2D,             # Grid object to access global arrays and range_factor
    f_neighbors::AbstractVector,    # (Not used by ClassicAlgorithm)
    df_neighbors::AbstractVector,   # Pre-gathered view of (f_j - f_i)
)::Real
    
    # Cast equation type to access velocity
    vel = (eq::LinearAdvection{2}).vel
    
    # --- 1. Get thread-local workspace, interpolator, and scaling factor ---
    thread_idx = Threads.threadid()
    ws = upwind.workspaces[thread_idx]
    interp = upwind.interpolator
    interpRange = pg.range_factor # Access the range factor

    # Get references to GLOBAL grid data arrays
    dx_all_full = pg.neighbor_xdistance
    dy_all_full = pg.neighbor_ydistance
    w_all_full = pg.neighbor_weights # Use pre-gathered weights

    num_nb = length(nb_slice)
    
    if num_nb == 0; return 0.0; end
    
    # Ensure the *internal* buffers are large enough
    ensure_capacity!(ws, num_nb)

    # --- 2. The Filter & Compact Loop ---
    # This loop filters neighbors based on the *unscaled* distances
    # but stores the *scaled* distances in the workspace.
    count = 0
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        dx_k_unscaled = dx_all_full[global_idx]
        dy_k_unscaled = dy_all_full[global_idx]

        # Upwind check uses unscaled distances
        tmp = dx_k_unscaled * vel[1]
        if tmp + dy_k_unscaled * vel[2] < 0
            count += 1
            # Store SCALED distances in the workspace
            ws.dxVec[count] = dx_k_unscaled / interpRange
            ws.dyVec[count] = dy_k_unscaled / interpRange
            # Store pre-gathered difference and weight
            ws.dfVec[count] = df_neighbors[local_idx] # df needs local index
            ws.wVec[count]  = w_all_full[global_idx]
        end
    end

    num_upwind = count
    if num_upwind < upwind.order; return 0.0; end
    
    ensure_capacity!(interp, num_upwind)

    # --- 3. Process the Compacted Data ---
    # These views now contain the SCALED dx/dy values
    dxVec_upwind = @view ws.dxVec[1:num_upwind]
    dyVec_upwind = @view ws.dyVec[1:num_upwind]
    dfVec_upwind = @view ws.dfVec[1:num_upwind]
    wVec_upwind  = @view ws.wVec[1:num_upwind]
    
    local res1, res2
    # The interpolator works with the scaled distances
    if upwind.order == 1
        res1, res2 = interp(dxVec_upwind, dyVec_upwind, wVec_upwind, dfVec_upwind)
    elseif upwind.order == 2
        res1, res2, res3, res4 = interp(dxVec_upwind, dyVec_upwind, wVec_upwind, dfVec_upwind)
        # Curvature cannot be set as `pg` is not an argument in this signature
    end
    
    # --- SCALE the final derivative result ---
    # res1 and res2 represent the scaled derivatives (d/d(x/L), d/d(y/L))
    # Divide by interpRange to get the actual derivatives (d/dx, d/dy)
    ddx = res1 / interpRange
    ddy = res2 / interpRange
    # --- END SCALE ---

    return (vel[1] * ddx + vel[2] * ddy)
end
"""
Functor for UpwindGradient (TiwariAlgorithm) with the 'fused' signature.
Calculates the upwind gradient for a single particle `i`.
Uses pre-gathered neighbor data, minimizes view creation, and uses pg.range_factor for scaling.
"""
function (upwind::UpwindGradient{2, <:UpwindWorkspaceTA, <:Any, TiwariAlgorithm})(
    eq::ScalarHyperbolicPDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid2D,             # Grid object to access global arrays and range_factor
    f_neighbors::AbstractVector,    # (Not used directly by Tiwari)
    df_neighbors::AbstractVector,   # Pre-gathered view of (f_j - f_i)
)::Real
    
    # Cast equation type to access velocity
    vel = (eq::LinearAdvection{2}).vel 
    
    # --- 1. Get thread-local workspace, interpolator, and scaling factor ---
    thread_idx = Threads.threadid()
    ws = upwind.workspaces[thread_idx] 
    interp = upwind.interpolator
    interpRange = pg.range_factor # Access the range factor from the grid

    # Get references to GLOBAL grid data arrays
    dx_all_full = pg.neighbor_xdistance
    dy_all_full = pg.neighbor_ydistance
    w_all_full = pg.neighbor_weights # Use pre-gathered weights

    num_neighbors = length(nb_slice)
    
    if num_neighbors < upwind.order 
        return 0.0 
    end
    
    ensure_capacity!(ws, num_neighbors) 

    # --- 2. Data Collection & Window Calculation (Single Pass) ---
    dxVec = @view ws.dxVec[1:num_neighbors]
    dyVec = @view ws.dyVec[1:num_neighbors]
    dfVec = @view ws.dfVec[1:num_neighbors]
    wVec  = @view ws.wVec[1:num_neighbors] 
    xWindow = @view ws.xWindow[1:num_neighbors] 
    yWindow = @view ws.yWindow[1:num_neighbors] 

    # Loop using enumerate for workspace index and nb_slice for global index
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        # --- SCALE dx, dy ---
        dx_k = dx_all_full[global_idx] / interpRange 
        dy_k = dy_all_full[global_idx] / interpRange
        # --- END SCALE ---
        
        dxVec[local_idx] = dx_k
        dyVec[local_idx] = dy_k
        dfVec[local_idx] = df_neighbors[local_idx] 
        wVec[local_idx]  = w_all_full[global_idx]   
        
        xWindow[local_idx] = (vel[1] * dx_k <= 0.0) 
        yWindow[local_idx] = (vel[2] * dy_k <= 0.0) 
    end

    ddx = 0.0
    ddy = 0.0
    
    # --- 3. X-Derivative Calculation ---
    stencil_size_x = sum(xWindow) 

    if stencil_size_x >= upwind.order
        partition_workspace!(xWindow, dxVec, dyVec, dfVec, wVec) 
 
        dx_stencil_x = @view dxVec[1:stencil_size_x]
        dy_stencil_x = @view dyVec[1:stencil_size_x]
        df_stencil_x = @view dfVec[1:stencil_size_x]
        w_stencil_x  = @view wVec[1:stencil_size_x] 

        ensure_capacity!(interp, stencil_size_x) 
        res_x = interp(dx_stencil_x, dy_stencil_x, w_stencil_x, df_stencil_x) 
        
        # --- SCALE result ---
        ddx = res_x[1] / interpRange
        # --- END SCALE ---
    end
        
    # --- 4. Y-Derivative Calculation ---
    stencil_size_y = sum(yWindow) 

    if stencil_size_y >= upwind.order
        partition_workspace!(yWindow, dxVec, dyVec, dfVec, wVec)

        dx_stencil_y = @view dxVec[1:stencil_size_y]
        dy_stencil_y = @view dyVec[1:stencil_size_y]
        df_stencil_y = @view dfVec[1:stencil_size_y]
        w_stencil_y  = @view wVec[1:stencil_size_y] 
        
        ensure_capacity!(interp, stencil_size_y) 
        res_y = interp(dx_stencil_y, dy_stencil_y, w_stencil_y, df_stencil_y) 

        # --- SCALE result ---
        ddy = res_y[2] / interpRange 
        # --- END SCALE ---
    end

    # --- 5. Final Result ---
    # The final combination doesn't need scaling because ddx/ddy are already scaled derivatives
    return ddx * vel[1] + ddy * vel[2] 
end


"""
Functor for UpwindGradient (PraveenAlgorithm) using minimal workspace
and precalculated weights.
"""
function (upwind::UpwindGradient{2, <:UpwindWorkspacePA, <:Any, PraveenAlgorithm})(
    eq::ScalarHyperbolicPDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid2D,             # Grid object to access global arrays and range_factor
    f_neighbors::AbstractVector,    # (Not used directly by Praveen)
    df_neighbors::AbstractVector,   # Pre-gathered view of (f_j - f_i)
)::Real
    
    vel = (eq::LinearAdvection{2}).vel 
    
    # --- 1. Get workspace, scaling factor, and refs to global data ---
    thread_idx = Threads.threadid()
    ws = upwind.workspaces[thread_idx]
    interpRange = pg.range_factor 

    dx_all_full = pg.neighbor_xdistance
    dy_all_full = pg.neighbor_ydistance
    w_all_full = pg.neighbor_weights # <-- USE PRECALCULATED WEIGHTS

    num_neighbors = length(nb_slice)
    
    if num_neighbors < 3; return 0.0; end 
    
    ensure_capacity!(ws, num_neighbors) 

    # --- 2. Least-Squares System Setup (Explicit Loop with Scaling) ---
    A11 = 0.0
    A22 = 0.0
    A12 = 0.0
    
    # Use workspace buffers directly
    coeff_x_Vec = ws.coeff_x_Vec
    coeff_y_Vec = ws.coeff_y_Vec
    cijVec      = ws.cijVec

    inv_interpRange = 1.0 / interpRange # Precompute inverse for scaling

    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        # Fetch precalculated weight
        w_k = w_all_full[global_idx]
        
        # Scale distances
        dx_k = dx_all_full[global_idx] * inv_interpRange 
        dy_k = dy_all_full[global_idx] * inv_interpRange
        
        # Accumulate matrix components
        A11 += w_k * dx_k * dx_k
        A22 += w_k * dy_k * dy_k
        A12 += w_k * dx_k * dy_k
    end
    
    D = A11 * A22 - A12^2
    if abs(D) < 1e-14; return 0.0; end
    invD = 1.0 / D

    # --- 3. Divergence Calculation (Explicit Loop) ---
    div = 0.0 # Accumulator for the final dot product

    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        # Fetch precalculated weight and scaled distances (can re-fetch or buffer)
        w_k  = w_all_full[global_idx] 
        dx_k = dx_all_full[global_idx] * inv_interpRange
        dy_k = dy_all_full[global_idx] * inv_interpRange
        
        # Solve for coefficients (no need to store in full Vecs if not reused)
        coeff_x = (w_k * (A22 * dx_k - A12 * dy_k)) * invD
        coeff_y = (w_k * (A11 * dy_k - A12 * dx_k)) * invD
        # Store coefficients temporarily if needed, otherwise use directly
        # coeff_x_Vec[local_idx] = coeff_x 
        # coeff_y_Vec[local_idx] = coeff_y

        # Rotational vectors
        hyp = hypot(dx_k, dy_k)
        nx, ny = if hyp < 1e-14
            (1.0, 0.0) # Handle dx=dy=0 case
        else
            invHyp = 1.0 / hyp
            (dx_k * invHyp, dy_k * invHyp)
        end
        sx = -ny 
        sy = nx  

        # Compute adapted coefficients
        alfaBar = nx * coeff_x + ny * coeff_y
        betaBar = sx * coeff_x + sy * coeff_y

        # Dot products with velocity
        vel_n = vel[1] * nx + vel[2] * ny
        vel_s = vel[1] * sx + vel[2] * sy

        # Calculate bracket terms
        bracketMinus1 = min(vel_n, 0.0)
        bracketMinus2 = min(betaBar * vel_s, 0.0)
    
        # Calculate final coefficient `cij`
        cij = alfaBar * bracketMinus1 + bracketMinus2
        # cijVec[local_idx] = cij # Only store if needed later

        # Accumulate dot product using precalculated df
        div += cij * df_neighbors[local_idx] 
    end
    
    # Scale the final result
    return 2 * div / interpRange
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

