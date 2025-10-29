export MUSCLORDER, MUSCLORDER1, MUSCLORDER2, MUSCLORDER3, MUSCLORDER4, AbstractSlopeLimiter, BarthJespersenLimiter, VenkatakrishnanLimiter, SuperbeeLimiter, MinmodLimiter, NoLimiter

abstract type MUSCLORDER end
struct MUSCLORDER1 <: MUSCLORDER end
struct MUSCLORDER2 <: MUSCLORDER end
struct MUSCLORDER3 <: MUSCLORDER end
struct MUSCLORDER4 <: MUSCLORDER end


abstract type AbstractSlopeLimiter end
abstract type RealSlopeLimiter <: AbstractSlopeLimiter end
struct BarthJespersenLimiter <: RealSlopeLimiter end
struct VenkatakrishnanLimiter <: RealSlopeLimiter end
struct SuperbeeLimiter <: RealSlopeLimiter end
struct MinmodLimiter <: RealSlopeLimiter end
struct NoLimiter <: AbstractSlopeLimiter end

# --- In Interpolations.jl, near your other interpolator definitions ---

# --- 1. Define MUSCL Workspace and Updated Struct ---
abstract type MUSCLWorkspace end

# --- In MUSCL.jl ---

# (Keep abstract types, limiter funcs, find_closest_lr_neighbors_1D)

# --- REFACTORED: 1D Workspace with flat buffers ---
mutable struct MUSCLWorkspace1D <: MUSCLWorkspace
    # --- FLATTENED per-interaction coefficient storage ---
    alfaijs::Vector{Float64}     # for 1st-order slope / 3rd-order div
    alfaij_bars::Vector{Float64} # for 2nd-order slope / 1st-order div
    betaijs::Vector{Float64}     # for 2nd-order curve / 2nd-order div
    gammaijs::Vector{Float64}    # for 4th-order

    # --- PER-PARTICLE derivative storage (already flat) ---
    slopes::Vector{Float64}
    curves_xx::Vector{Float64}
    # (curves_yy is not needed for 1D)

    # --- Temporary buffers for a single particle's calculation ---
    # (These are small and can be re-used by each thread)
    dx_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    A_buffer::Matrix{Float64} # For pinv
    
    max_neighbors_local::Int # Max neighbors for a single particle

    function MUSCLWorkspace1D(
        initial_particle_cap::Int = 100,
        initial_neighbor_cap::Int = 20, # Max neighbors for temp buffers
        initial_flat_cap::Int = 1000  # Capacity for total interactions
    )
        new(
            zeros(initial_flat_cap), zeros(initial_flat_cap), # alfaijs, alfaij_bars
            zeros(initial_flat_cap), zeros(initial_flat_cap), # betaijs, gammaijs
            zeros(initial_particle_cap), zeros(initial_particle_cap), # slopes, curves_xx
            zeros(initial_neighbor_cap), zeros(initial_neighbor_cap), # dx_buffer, w_buffer
            zeros(initial_neighbor_cap, 4), # A_buffer
            initial_neighbor_cap
        )
    end
end

# --- NEW: 2D Workspaces split by order ---
abstract type MUSCLWorkspace2D <: MUSCLWorkspace end

"""
Workspace for 2D, 1st Order MUSCL.
Contains flat buffers for coefficients and per-particle slope storage.
"""
struct MUSCLWorkspace2D1O <: MUSCLWorkspace2D
    # --- FLATTENED per-interaction coefficient storage ---
    alfaijs::Vector{Float64}
    betaijs::Vector{Float64}

    # --- PER-PARTICLE slope storage (already flat) ---
    slopes_x::Vector{Float64}
    
    slopes_y::Vector{Float64}

    function MUSCLWorkspace2D1O(
        initial_particle_cap::Int = 100, 
        initial_flat_cap::Int = 1000 # Capacity for total interactions
    )
    
new(
            zeros(initial_flat_cap), zeros(initial_flat_cap), # alfaijs, betaijs
            zeros(initial_particle_cap), zeros(initial_particle_cap) # slopes_x, slopes_y
        )
    end
end

"""
Workspace for 2D, 2nd Order MUSCL.
Contains extended flat buffers for coefficients, per-particle derivative storage,
and a temporary matrix buffer for the pseudo-inverse calculation.
"""
struct MUSCLWorkspace2D2O <: MUSCLWorkspace2D
    # --- FLATTENED per-interaction coefficient storage ---
    alfaijs::Vector{Float64}     # for fx
    betaijs::Vector{Float64}     # for fy
    alfaij_bars::Vector{Float64} # for fxx
    betaij_bars::Vector{Float64} # for fyy
    gammaijs::Vector{Float64}    # for fxy

    # --- PER-PARTICLE derivative storage (already flat) ---
    slopes_x::Vector{Float64}
    slopes_y::Vector{Float64}
    curves_xx::Vector{Float64} 
    curves_yy::Vector{Float64} 
    curves_xy::Vector{Float64} 

    function MUSCLWorkspace2D2O(
        initial_particle_cap::Int = 100, 
        initial_neighbor_cap::Int = 20, # Max neighbors for temp A_buffer
        initial_flat_cap::Int = 1000 # Capacity for total interactions
    )
        
        # Create one A_buffer for each thread
        n_threads = Threads.nthreads()
        thread_buffers = [
            zeros(Float64, initial_neighbor_cap, 5) for _ in 1:n_threads
        ]

new(
            zeros(initial_flat_cap), zeros(initial_flat_cap), # alfaijs, betaijs
            zeros(initial_flat_cap), zeros(initial_flat_cap), # alfaij_bars, betaij_bars
            zeros(initial_flat_cap), # gammaijs
            zeros(initial_particle_cap), zeros(initial_particle_cap), # slopes_x, slopes_y
            zeros(initial_particle_cap), zeros(initial_particle_cap), # curves_xx, curves_yy
            zeros(initial_particle_cap), # curves_xy
        )
    end
end

"""
Sets all temporary buffers and coefficient storage in the MUSCL workspace to zero.
"""
function zero_workspace!(ws::MUSCLWorkspace1D)
    # Zero out temporary buffers
    fill!(ws.dx_buffer, 0.0)
    fill!(ws.w_buffer, 0.0)
    fill!(ws.A_buffer, 0.0)

    # Clear out all previously calculated coefficients
    for p_idx in 1:length(ws.alfaijs)
        empty!(ws.alfaijs[p_idx])
        empty!(ws.alfaij_bars[p_idx])
        empty!(ws.betaijs[p_idx])
        empty!(ws.gammaijs[p_idx])
    end
end
function ensure_capacity!(ws::MUSCLWorkspace1D, n::Int)
    # Check if the required number of neighbors `n` exceeds the current buffer capacity.
    
    if n > length(ws.dx_buffer)
        # Calculate a new capacity with a 25% buffer to avoid frequent re-allocations.
        new_capacity = n + n ÷ 4
        
        # Resize all vectors at once using broadcasting.
        resize!.((ws.dx_buffer, ws.w_buffer), new_capacity)
        
        # Re-create the matrix buffer with the new size.
      
        # Using `undef` is slightly faster than `zeros` if it's always overwritten.
        ws.A_buffer = Matrix{Float64}(undef, new_capacity, 4)
    end
    return nothing
end

function _ensure_coeff_vectors_sized!(ws::MUSCLWorkspace1D, p_idx::Int, n::Int)
    # Check if the current buffer for this particle is too small
    if length(ws.alfaijs[p_idx]) < n
        # Calculate a new capacity with a 25% buffer
        new_capacity = n #+ n ÷ 4
        
   
     # Resize all coefficient vectors for this particle at once
        resize!.((
            ws.alfaijs[p_idx], 
            ws.alfaij_bars[p_idx], 
            ws.betaijs[p_idx], 
            ws.gammaijs[p_idx]
        ), new_capacity)
    end
    return nothing
end

function ensure_particle_capacity!(ws::MUSCLWorkspace1D, N::Int)
    current_size = length(ws.alfaijs)
 
   if current_size < N
        new_capacity = N #+ N ÷ 4
        num_to_add = new_capacity - current_size
        
        # Grow the outer vector of vectors
        for _ in 1:num_to_add
            push!(ws.alfaijs, Float64[])
            push!(ws.alfaij_bars, Float64[])
         
   push!(ws.betaijs, Float64[])
            push!(ws.gammaijs, Float64[])
        end
        
        # Resize the simple vector to the new capacity
        resize!(ws.slopes, new_capacity)
        resize!(ws.curves_xx, new_capacity)
        resize!(ws.curves_yy, new_capacity)
    end
    return nothing
end

# --- NEW: ensure_particle_capacity! for 2D workspaces ---
function ensure_particle_capacity!(ws::MUSCLWorkspace2D1O, N::Int)
    if length(ws.slopes_x) < N
        resize!.((
            ws.slopes_x, ws.slopes_y
        ), N)
    end
    return nothing
end

function ensure_particle_capacity!(ws::MUSCLWorkspace2D2O, N::Int)
    if length(ws.slopes_x) < N
        resize!.((
            ws.slopes_x, ws.slopes_y,
            ws.curves_xx, ws.curves_yy, ws.curves_xy
        ), N)
    end
    return nothing
end


# --- NEW: ensure_coefficients_capacity! for 2D workspaces ---
"""
Ensures the flat coefficient arrays can hold data for every neighbor interaction.
"""
function ensure_coefficients_capacity!(ws::MUSCLWorkspace2D1O, grid::ParticleGrid2D{S}) where S
    required_len = length(grid.neighbor_indices)
    if length(ws.alfaijs) < required_len
   
     # Resize all flat coefficient arrays at once
        new_capacity = required_len + required_len ÷ 4
        resize!.((
            ws.alfaijs, ws.betaijs
        ), new_capacity)
    end
    return nothing
end

function ensure_coefficients_capacity!(ws::MUSCLWorkspace2D2O, grid::ParticleGrid2D{S}) where S
    required_len = length(grid.neighbor_indices)
    if length(ws.alfaijs) < required_len
   
     # Resize all flat coefficient arrays at once
        new_capacity = required_len + required_len ÷ 4
        resize!.((
            ws.alfaijs, ws.betaijs, 
            ws.alfaij_bars, ws.betaij_bars, 
            ws.gammaijs
        ), new_capacity)
    end
    return nothing
end


# --- NEW: ensure_capacity! for 2D (temporary buffers) ---
# This handles the temporary arrays used for a single particle's neighbors.
# For 1st Order, no temp buffers are needed for initGI!
function ensure_capacity!(ws::MUSCLWorkspace2D1O, n::Int)
    return nothing # No temp buffers to resize
end


struct MUSCL{D,ORDER<:MUSCLORDER, L<:AbstractSlopeLimiter, NFF <: NumericalFluxFunction, WS<:MUSCLWorkspace} <: GradientInterpolator
    order::ORDER
    limiter::L
    res::Vector{Float64}
    numericalFlux::NFF
    workspace::WS
end

# --- NEW: Updated MUSCL Constructor ---
function MUSCL(
    order::Int, 
   
 dimension::Int; 
    limiter::L=NoLimiter(), 
    numericalFlux=RusanovFlux()
) where {L<:AbstractSlopeLimiter}

    # --- Workspace Selection ---
    if dimension == 1
        ws = MUSCLWorkspace1D()

    elseif dimension == 2
        if order == 1
            ws = MUSCLWorkspace2D1O()
        elseif order == 2
            ws = MUSCLWorkspace2D2O()
        else
             # Default to 2O for orders 3, 4 for now, or error
            error("Order $order not fully supported for 2D workspace yet.")
            # ws = MUSCLWorkspace2D2O() 
        end
    else
        error("Dimension $dimension not supported.")
    end
    
    res_size = (dimension == 1) ?
order : (order == 1 ? 2 : 5) # Determine size of result buffer
    
    WS = typeof(ws)
    
if order == 1
        return MUSCL{dimension, MUSCLORDER1, L, typeof(numericalFlux), WS}(MUSCLORDER1(), limiter, zeros(res_size), numericalFlux, ws)
    elseif order == 2
        return MUSCL{dimension, MUSCLORDER2, L, typeof(numericalFlux), WS}(MUSCLORDER2(), limiter, zeros(res_size), numericalFlux, ws)
    elseif order == 3
        return MUSCL{dimension, MUSCLORDER3, L, typeof(numericalFlux), WS}(MUSCLORDER3(), limiter, zeros(res_size), numericalFlux, ws)
    elseif order == 4
        return MUSCL{dimension, MUSCLORDER4, L, typeof(numericalFlux), WS}(MUSCLORDER4(), limiter, zeros(res_size), numericalFlux, ws)
    else
        error("Order must be 1, 2, 3, or 4.")
    end
end

"""
    initGIBuffers!(g::MUSCL, pg)

A high-level wrapper that ensures all buffers within the MUSCL gradient
interpolator's workspace are adequately sized for the given particle grid.

This function automatically dispatches to the correct implementation based 
on the type of `g.workspace`.
"""
function initGIBuffers!(g::MUSCL, pg)
    # Dispatch to the specific implementation based on the workspace type
    initGIBuffers!(g.workspace, pg)
    return nothing
end

"""
(2D, Order 1 Implementation) Ensures 2D1O workspace buffers are sized.
- Resizes per-particle arrays (slopes_x, slopes_y) to size N[cite: 21].
- Resizes flat coefficient arrays (alfaijs, betaijs) based on the 
  total number of interactions, plus a 25% buffer[cite: 22].
"""
function initGIBuffers!(ws::MUSCLWorkspace, pg::ParticleGrid)
    N = pg.N
    
    # 1. Ensure capacity for per-particle buffers (size N)
    ensure_particle_capacity!(ws, N)
    
    # 2. Ensure capacity for flat coefficient buffers (size M + 25%)
    # This existing function already adds the 25% buffer [cite: 22]
    ensure_coefficients_capacity!(ws, pg)

    # 3. Temporary buffers: Not needed for 2D1O [cite: 24]
    
    return nothing
end

# --- NEW: Localized Helper Functions for initGI! (2D, Order 1) ---


include("MUSCLCoeffs.jl")
include("MUSCLUtils.jl")
include("MUSCLLimiter.jl")

# --- In MUSCL.jl, replace the 1D initGI! function ---

"""
    initGI!(muscl::MUSCL{1, ORDER}, ...)

(1D Main) Orchestrator function to calculate and store derivatives.
Dispatches calculation based on muscl.order.
"""
function initGI!(
    muscl::MUSCL{1, ORDER},
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    pg::ParticleGrid,               # Grid object (will be 1D)
    neighbor_fs::AbstractVector,    # The flat neighbor-value buffer
    neighbor_dfs::AbstractVector    # The flat neighbor-difference buffer
) where {ORDER <: MUSCLORDER}
    
    ws = muscl.workspace::MUSCLWorkspace1D
    nb_slice = getNBSlice(pg, i)
    num_nb = length(nb_slice)

    # --- Handle zero-neighbor case ---
    if num_nb == 0
        _zero_coeffs_1d!(nb_slice, ws) # Zero coefficients
        # Calculate returns 0.0 or (0.0, 0.0)
        slope_x, _ = _calculate_slopes(muscl.order, nb_slice, neighbor_dfs, ws) 
        higher_derivatives = _calculate_higher_derivatives(muscl.order, nb_slice, neighbor_dfs, ws)
        # Save zero derivatives
        _save_derivatives!(ws, i, 0.0, higher_derivatives) 
        return
    end

    # Get refs to global buffers needed by helpers
    dx = pg.neighbor_xdistance
    w  = pg.neighbor_weights
    
    # --- 1. Find Scaling Factor L ---
    L = 1e-14
    @inbounds for k in nb_slice
        L = max(L, abs(dx[k]))
    end

    # --- 2. Compute Coefficients ---
    # Dispatches based on muscl.order
    _compute_coeffs!(
        muscl.order, nb_slice, dx, w, L,
        ws.alfaijs, ws.alfaij_bars, ws.betaijs, ws.gammaijs
    )

    # --- 3. Calculate Slopes (and potentially Curve for ORDER2+) ---
    # Dispatches based on muscl.order
    slope_x, curve_for_limit = _calculate_slopes(muscl.order, nb_slice, neighbor_dfs, ws)

    # --- 4. Limit Slopes ---
    # Dispatches based on muscl.limiter type
    # (Uses the _limit_slopes helper functions, requires renaming _limit_slopes_1d)
    slope_x = _limit_slopes(muscl.limiter, slope_x, i, f_i, pg, neighbor_fs)

    # --- 5. Calculate Higher Derivatives (if any) ---
    # Dispatches based on muscl.order
    higher_derivatives = _calculate_higher_derivatives(muscl.order, nb_slice, neighbor_dfs, ws)

    # --- 6. Store Final Derivatives ---
    if isnan(slope_x) # Basic NaN check
        error("Found NaN while calculating derivatives for particle $i!")
    end
    # Add checks for higher_derivatives if needed (e.g., check curve_xx)

    # Dispatches based on muscl.order implicitly via tuple type
    _save_derivatives!(ws, i, slope_x, higher_derivatives)
    return
end

# --- In MUSCL.jl, replace the 2D initGI! functions ---

"""
    initGI!(muscl::MUSCL{2}, ...)

(2D Main) Orchestrator function to calculate and store derivatives.
Dispatches calculation based on muscl.order and ws type.
"""
function initGI!(
    muscl::MUSCL{2},
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    pg::ParticleGrid,               # Grid object (will be 2D)
    neighbor_fs::AbstractVector,    # The flat neighbor-value buffer
    neighbor_dfs::AbstractVector    # The flat neighbor-difference buffer
)
    ws = muscl.workspace # ws will be MUSCLWorkspace2D1O or MUSCLWorkspace2D2O
    nb_slice = getNBSlice(pg, i)
    num_nb = length(nb_slice)

    # --- Handle zero-neighbor case ---
    if num_nb == 0
        _zero_coeffs!(nb_slice, ws) # Zero coefficients
        higher_derivatives = _calculate_higher_derivatives(muscl.order, nb_slice, neighbor_dfs, ws) # Returns () or (0,0,0)
        _save_derivatives!(ws, i, 0.0, 0.0, higher_derivatives) # Save zero derivatives
        return
    end

    # Get refs to global buffers needed by helpers
    dx = pg.neighbor_xdistance
    dy = pg.neighbor_ydistance
    w  = pg.neighbor_weights

    # --- 1. Compute Coefficients ---
    # Dispatches based on muscl.order AND ws type implicitly
    _compute_coeffs!(muscl.order, nb_slice, ws, dx, dy, w)

    # --- 2. Calculate Slopes ---
    # (Uses the existing _calculate_slopes helper, which expects alfaij/betaij views)
    # NOTE: We need views here, matching the required arguments
    alfaij = ws.alfaijs
    betaij = ws.betaijs
    slope_x, slope_y = _calculate_slopes(nb_slice, neighbor_dfs, alfaij, betaij)

    # --- 3. Limit Slopes ---
    # (Uses the existing _limit_slopes helpers)
    slope_x, slope_y = _limit_slopes(muscl.limiter, slope_x, slope_y, nb_slice, f_i, neighbor_fs, dx, dy)

    # --- 4. Calculate Higher Derivatives ---
    # Dispatches based on muscl.order and ws type
    higher_derivatives = _calculate_higher_derivatives(muscl.order, nb_slice, neighbor_dfs, ws)

    # --- 5. Store Final Derivatives ---
    if isnan(slope_y) || isnan(slope_x) # Basic NaN check
        error("Found NaN while calculating derivatives for particle $i!")
    end
    # Add checks for higher_derivatives if needed

    # Dispatches based on ws type
    _save_derivatives!(ws, i, slope_x, slope_y, higher_derivatives)
    return
end

# --- 4. Refactored 1D MUSCL Functor ---
function (muscl::MUSCL{D,ORDER,L})(
    particleGrid::ParticleGrid1D, 
    particleIndex::Integer, 
    fVec::AbstractVector{<:Real}, 
    eq::ScalarHyperbolicPDE, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real where {D,ORDER<:MUSCLORDER,L<:AbstractSlopeLimiter}
    
    div = 0.0
    ws = muscl.workspace
    neighbors_i = particleGrid.neighbour_indices[particleIndex]
    
    # Retrieve the correct coefficients for the divergence sum based on order
    div_coeffs = (ORDER == MUSCLORDER1) ? ws.alfaijs[particleIndex] : ws.alfaij_bars[particleIndex]

    for (index_in_list, nbIndex) in enumerate(neighbors_i)
        deltaPos = getDistance(particleGrid, particleIndex, nbIndex)
        
        fij, fji = reconstruct_interface_states(muscl.order, particleGrid, fVec, ws, particleIndex, nbIndex, deltaPos)
        
        fm, fp = sortFlux(fij, fji, deltaPos)
        div += div_coeffs[index_in_list] * (muscl.numericalFlux(fm, fp, eq) - flux(eq, fVec[particleIndex]))
    end

    if setCurvature
 
       if ORDER == MUSCLORDER1
            particleGrid.curvatures[particleIndex] = 0.0
        else
            betaij_i = ws.betaijs[particleIndex]
            particleGrid.curvatures[particleIndex] = sum(betaij_i[k]*(fVec[nb_k] - fVec[particleIndex]) for (k, nb_k) in enumerate(neighbors_i))
        end
    end
    
    return 2 * div
end

# --- NEW: Localized 2D Functor ---
"""
    (muscl::MUSCL{2, ORDER})(...)

Calculates the divergence for a single particle `i` using pre-calculated slopes
and neighbor data provided as views.
"""
function (muscl::MUSCL{2, ORDER})(
    eq::ScalarHyperbolicPDE,
    i::Int,                         # Current particle index
    f_i::Real,
    neighbor_slice::UnitRange{Int},                      # Value of f at particle i
    pg::ParticleGrid,
    f_neighbors::AbstractVector,    # View of neighbor f-values
    df_neighbors::AbstractVector,   # View of neighbor df-values
)::Real where {ORDER<:MUSCLORDER}
    
    div = 0.0
    ws = muscl.workspace
    nFlux = muscl.numericalFlux
    
    if pg.num_neighbors[i] == 0; return 0.0; end

    dx = pg.neighbor_xdistance
    dy = pg.neighbor_ydistance
    nb_indices = pg.neighbor_indices

    fx, fy = flux(eq, f_i)
    # Loop over neighbors using the local index `k_local`
    for k_global in neighbor_slice
        # Get global index for coefficient arrays
        
        # Get data from views
        nbIndex = nb_indices[k_global]
        deltaX = dx[k_global]
        deltaY = dy[k_global]
        f_j = f_neighbors[k_global]
        
        # Get pre-calculated coefficients
        alfaij = ws.alfaijs[k_global]
  
        betaij = ws.betaijs[k_global]
        
        # This call uses pre-calculated slopes from the workspace
        fij, fji = reconstruct_interface_states(muscl.order, ws, f_i, f_j, i, nbIndex, deltaX, deltaY)

        fmx, fpx, fmy, fpy = sortFlux(fij, fji, deltaX, deltaY)
        
        div += alfaij * (nFlux(fmx, fpx, eq, 1) - fx) + 

               betaij * (nFlux(fmy, fpy, eq, 2) - fy)
    end
    
    return 2 * div
end
