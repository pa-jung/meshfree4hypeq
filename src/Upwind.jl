# ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  # Only relevant in 2D. In 1D, all algorithms are the same.
abstract type TiwariAlgorithm <: UpwindAlgorithm end  # Split domain in left and right for d/dx, and up and down for d/dy.
abstract type PraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type ClassicAlgorithm <: UpwindAlgorithm end  # Take all points 'behind' center point. 
abstract type RusanovAlgorithm <: UpwindAlgorithm end # This is no upwinding of course but easy implementation in this framework (numerical Flux given does not have to be upwind)

abstract type UpwindWorkspace end

using InteractiveUtils


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
struct UpwindGradient{D, WS <: UpwindWorkspace, I <: Interpolator, Algorithm <: UpwindAlgorithm} <: GradientInterpolator
    order::Int
    weightFunction::MLSWeightFunction
    numericalFlux::NumericalFluxFunction
    workspaces::Vector{WS}
    interpolator::I
    # --- Modify the UpwindGradient Constructor ---
    function UpwindGradient(order, dimension; numericalFlux::NumericalFluxFunction=UpwindFlux(), algType::String="Classic", weightFunction::MLSWeightFunction=exponentialWeightFunction())
        @assert order >= 1 "Order must be larger or equal to one."
        @assert algType in ["Classic", "Tiwari", "Praveen", "NonLinearPraveen"]
        
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
        else
            error("Algorithm type $algType not fully configured for workspace selection.")
        end

        n_threads = Threads.nthreads()
        println(n_threads)
        workspaces = [WS_eltype(100) for _ in 1:n_threads] 

        interpolator = Interpolator{dimension, order, 1}()
        I = typeof(interpolator)

        new{dimension, WS_eltype, I, alg_type}(order, weightFunction, numericalFlux, workspaces, interpolator)
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
    thread_idx = mod1(Threads.threadid(),Threads.nthreads())
    ws = upwind.workspaces[thread_idx]
    interp = upwind.interpolator

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
    for global_idx in nb_slice
        dx_k_unscaled = dx_all_full[global_idx]
        dy_k_unscaled = dy_all_full[global_idx]

        # Upwind check uses unscaled distances
        tmp = dx_k_unscaled * vel[1]
        if tmp + dy_k_unscaled * vel[2] < 0
            count += 1
            # Store SCALED distances in the workspace
            ws.dxVec[count] = dx_k_unscaled
            ws.dyVec[count] = dy_k_unscaled
            # Store pre-gathered difference and weight
            ws.dfVec[count] = df_neighbors[global_idx] # df needs local index
            ws.wVec[count]  = w_all_full[global_idx]
        end
    end

    num_upwind = count

    if num_upwind < upwind.order; return 0.0; end
    ensure_capacity!(interp, num_upwind)
    
    local res1, res2
    # The interpolator works with the scaled distances
    if upwind.order == 1
        res1, res2 = interp(ws.dxVec, ws.dyVec, ws.wVec,  ws.dfVec, num_upwind)
    elseif upwind.order == 2
        res1, res2, _, _, _ = interp(ws.dxVec, ws.dyVec,  ws.wVec, ws.dfVec, num_upwind)
        # Curvature cannot be set as `pg` is not an argument in this signature
    end
    
    # --- SCALE the final derivative result ---
    # res1 and res2 represent the scaled derivatives (d/d(x/L), d/d(y/L))
    # Divide by interpRange to get the actual derivatives (d/dx, d/dy)
    ddx = res1
    ddy = res2
    # --- END SCALE ---
    #@assert ddx < 1e-5 "div Non zero $(vel[1] * ddx + vel[2] * ddy)"
    return (vel[1] * ddx + vel[2] * ddy)
end

function (upwind::UpwindGradient{2, <:UpwindWorkspaceTA, <:Any, TiwariAlgorithm})(
    eq::ScalarHyperbolicPDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid2D,             # Grid object
    f_neighbors::AbstractVector,    # (Not used)
    df_neighbors::AbstractVector,   # Pre-gathered diffs
)::Real
    
    vel = (eq::LinearAdvection{2}).vel 
    
    # --- 1. Get workspace, interpolator, and global refs ---
    thread_idx = Threads.threadid()
    ws = upwind.workspaces[thread_idx] 
    interp = upwind.interpolator

    dx_all_full = pg.neighbor_xdistance
    dy_all_full = pg.neighbor_ydistance
    w_all_full = pg.neighbor_weights 

    num_neighbors = length(nb_slice)
    
    if num_neighbors < upwind.order 
         return 0.0 
    end
    
    ensure_capacity!(ws, num_neighbors) 
    
    # Get local handles to workspace buffers
    dxVec = ws.dxVec
    dyVec = ws.dyVec
    dfVec = ws.dfVec
    wVec  = ws.wVec 

    ddx = 0.0
    ddy = 0.0
    
    # --- 3. X-Derivative Calculation (Filter & Compact) ---
    stencil_size_x = 0 # Counter for x-stencil
    
    @inbounds for global_idx in nb_slice
        dx_k = dx_all_full[global_idx]
        
        # X-Upwind check
        if (vel[1] * dx_k <= 0.0) 
            stencil_size_x += 1
            dy_k = dy_all_full[global_idx]
            
            # Compact data into workspace
            dxVec[stencil_size_x] = dx_k
            dyVec[stencil_size_x] = dy_k
            dfVec[stencil_size_x] = df_neighbors[global_idx] # <-- Use global_idx
            wVec[stencil_size_x]  = w_all_full[global_idx]
        end
    end

    if stencil_size_x >= upwind.order
        ensure_capacity!(interp, stencil_size_x) 
        # Call interpolator with workspace arrays and the calculated stencil size
        res_x = interp(dxVec, dyVec, wVec, dfVec, stencil_size_x) 
        
        # No scaling on result
        ddx = res_x[1] 
    end
        
    # --- 4. Y-Derivative Calculation (Filter & Compact) ---
    stencil_size_y = 0 # Counter for y-stencil

    @inbounds for global_idx in nb_slice
        dy_k = dy_all_full[global_idx]

        # Y-Upwind check
        if (vel[2] * dy_k <= 0.0)
            stencil_size_y += 1
            dx_k = dx_all_full[global_idx]
            
            # Compact data into workspace (overwrites X-data, which is fine)
            dxVec[stencil_size_y] = dx_k
            dyVec[stencil_size_y] = dy_k
            dfVec[stencil_size_y] = df_neighbors[global_idx] # <-- Use global_idx
            wVec[stencil_size_y]  = w_all_full[global_idx]
        end
    end
    
    if stencil_size_y >= upwind.order
        ensure_capacity!(interp, stencil_size_y) 
        # Call interpolator with workspace arrays and the calculated stencil size
        res_y = interp(dxVec, dyVec, wVec, dfVec, stencil_size_y) 

        # No scaling on result
        ddy = res_y[2] 
    end

    # --- 5. Final Result ---
    return ddx * vel[1] + ddy * vel[2] 
end

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

    @inbounds for global_idx in nb_slice
        # Fetch precalculated weight
        w_k = w_all_full[global_idx]
        
        # Scale distances
        dx_k = dx_all_full[global_idx]
        dy_k = dy_all_full[global_idx]
        
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

    @inbounds for global_idx in nb_slice
        # Fetch precalculated weight and scaled distances (can re-fetch or buffer)
        w_k  = w_all_full[global_idx] 
        dx_k = dx_all_full[global_idx]
        dy_k = dy_all_full[global_idx]
        
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
        div += cij * df_neighbors[global_idx] 
    end
    
    # Scale the final result
    return 2 * div #/ interpRange
end

