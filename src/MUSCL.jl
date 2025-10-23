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

"""
    find_closest_lr_neighbors_1D(particleGrid, p_idx, fVec)

Finds the closest neighbor to the left and right of a given particle `p_idx`.
Optimized to work with the "Struct of Arrays" grid layout.
# Returns
- `(val_L, dist_L, val_R, dist_R)`: The solution value and signed distance for the
  closest left and right neighbors.
Returns `0.0` for values and distances if a 
  neighbor is not found on a given side.
"""
function find_closest_lr_neighbors_1D(
    particleGrid::ParticleGrid1D, 
    p_idx::Integer, 
    fVec::AbstractVector
)
    # Initialize return values
    val_L, dist_L = 0.0, 0.0
    val_R, dist_R = 0.0, 0.0
    
    # Initialize minimum distances found so far
    min_abs_dist_L = Inf
    min_dist_R = Inf

    # Access the neighbor list directly from the grid's SoA field
    for nb_idx in particleGrid.neighbour_indices[p_idx]
        dx_ij = getDistance(particleGrid, p_idx, nb_idx)

   
     if dx_ij > 1e-9 # Potential right neighbor
            if dx_ij < min_dist_R
                min_dist_R = dx_ij
                val_R = fVec[nb_idx]
                dist_R = dx_ij
            end
       
 elseif dx_ij < -1e-9 # Potential left neighbor
            abs_dx_ij = abs(dx_ij)
            if abs_dx_ij < min_abs_dist_L
                min_abs_dist_L = abs_dx_ij
                val_L = fVec[nb_idx]
                dist_L = dx_ij # Keep its negative sign
    
        end
        end
    end
    
    return val_L, dist_L, val_R, dist_R
end

# --- In Interpolations.jl, near your other interpolator definitions ---

# --- 1. Define MUSCL Workspace and Updated Struct ---
abstract type MUSCLWorkspace end

mutable struct MUSCLWorkspace1D <: MUSCLWorkspace
    # --- Per-particle coefficient storage (ragged arrays) ---
    alfaijs::Vector{Vector{Float64}}
    alfaij_bars::Vector{Vector{Float64}}
    betaijs::Vector{Vector{Float64}}
    gammaijs::Vector{Vector{Float64}}

    # Slopes Buffer
    slopes::Vector{Float64}
    curves_xx::Vector{Float64}
  
  curves_yy::Vector{Float64}

    # --- Temporary buffers for neighbor-specific calculations ---
    dx_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    A_buffer::Matrix{Float64}
    
    max_neighbors::Int

    function MUSCLWorkspace1D(N_particles::Int = 1, initial_capacity::Int=20)
        new(
            [Float64[] for _ in 1:N_particles], [Float64[] for _ in 1:N_particles],
            [Float64[] for _ in 1:N_particles], [Float64[] for _ in 1:N_particles],
         
   Vector{Float64}(undef,N_particles), zeros(N_particles), zeros(N_particles),
            zeros(Float64, initial_capacity), zeros(Float64, initial_capacity),
            zeros(Float64, initial_capacity, 4), initial_capacity
        )
    end
end

# --- NEW: 2D Workspaces split by order ---
abstract type MUSCLWorkspace2D <: MUSCLWorkspace end

"""
Workspace for 2D, 1st Order MUSCL.
Contains flat buffers for coefficients and per-particle slope storage.
"""
mutable struct MUSCLWorkspace2D1O <: MUSCLWorkspace2D
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
mutable struct MUSCLWorkspace2D2O <: MUSCLWorkspace2D
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

# --- Temporary buffers for a single particle's neighbors ---
    # This is now a vector of matrices, one for each thread.
    A_buffers::Vector{Matrix{Float64}}

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
        thread_buffers # A_buffers
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

# For 2nd Order, we need to resize all thread-local A_buffers
function ensure_capacity!(ws::MUSCLWorkspace2D2O, n::Int)
    # Check the size of the first buffer (all should be the same)
    if n > size(ws.A_buffers[1], 1)
        new_capacity = n + n ÷ 4
        
        # Resize all matrices in the vector
        for i in 1:length(ws.A_buffers)
            ws.A_buffers[i] = Matrix{Float64}(undef, new_capacity, 5)
        end
    end
    return nothing
end


struct MUSCL{D,ORDER<:MUSCLORDER, L<:AbstractSlopeLimiter, WF <: MLSWeightFunction, NFF <: NumericalFluxFunction, WS<:MUSCLWorkspace} <: GradientInterpolator
    order::ORDER
    limiter::L
    res::Vector{Float64}
    weightFunction::WF
    numericalFlux::NFF
    workspace::WS
end

# --- NEW: Updated MUSCL Constructor ---
function MUSCL(
    order::Int, 
   
 dimension::Int; 
    limiter::L=NoLimiter(), 
    weightFunction=exponentialWeightFunction(), 
    numericalFlux=RusanovFlux()
) where {L<:AbstractSlopeLimiter}
    
if !(limiter isa NoLimiter)
        @assert order == 1 "Slope limiting is currently only implemented for MUSCL order 1."
end

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
        return MUSCL{dimension, MUSCLORDER1, L, typeof(weightFunction), typeof(numericalFlux), WS}(MUSCLORDER1(), limiter, zeros(res_size), weightFunction, numericalFlux, ws)
    elseif order == 2
        return MUSCL{dimension, MUSCLORDER2, L, typeof(weightFunction), typeof(numericalFlux), WS}(MUSCLORDER2(), limiter, zeros(res_size), weightFunction, numericalFlux, ws)
    elseif order == 3
        return MUSCL{dimension, MUSCLORDER3, L, typeof(weightFunction), typeof(numericalFlux), WS}(MUSCLORDER3(), limiter, zeros(res_size), weightFunction, numericalFlux, ws)
    elseif order == 4
        return MUSCL{dimension, MUSCLORDER4, L, typeof(weightFunction), typeof(numericalFlux), WS}(MUSCLORDER4(), limiter, zeros(res_size), weightFunction, numericalFlux, ws)
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
(1D Implementation) Ensures 1D workspace buffers are sized.
- Resizes per-particle arrays (slopes, etc.) to size N[cite: 19].
- Resizes temporary buffers (dx_buffer, etc.) based on the maximum 
  number of neighbors any particle has, plus a 25% buffer[cite: 16].
"""
function initGIBuffers!(ws::MUSCLWorkspace1D, pg::ParticleGrid1D)
    N = pg.N
    
    # 1. Ensure capacity for per-particle buffers (size N)
    ensure_particle_capacity!(ws, N)
    
    # 2. Find the maximum number of neighbors for temporary buffers
    # (This can be a bit slow if N is huge, but only happens once)
    max_nb = 0
    if !isempty(pg.neighbour_indices)
        max_nb = maximum(length, pg.neighbour_indices)
    end
    
    # 3. Ensure capacity for temporary buffers (size max_nb + 25%)
    # This existing function already adds the 25% buffer [cite: 16]
    ensure_capacity!(ws, max_nb)
    
    return nothing
end

"""
(2D, Order 1 Implementation) Ensures 2D1O workspace buffers are sized.
- Resizes per-particle arrays (slopes_x, slopes_y) to size N[cite: 21].
- Resizes flat coefficient arrays (alfaijs, betaijs) based on the 
  total number of interactions, plus a 25% buffer[cite: 22].
"""
function initGIBuffers!(ws::MUSCLWorkspace2D1O, pg::ParticleGrid2D)
    N = length(pg.num_neighbors)
    
    # 1. Ensure capacity for per-particle buffers (size N)
    ensure_particle_capacity!(ws, N)
    
    # 2. Ensure capacity for flat coefficient buffers (size M + 25%)
    # This existing function already adds the 25% buffer [cite: 22]
    ensure_coefficients_capacity!(ws, pg)

    # 3. Temporary buffers: Not needed for 2D1O [cite: 24]
    
    return nothing
end

"""
(2D, Order 2 Implementation) Ensures 2D2O workspace buffers are sized.
- Resizes per-particle arrays (slopes_x, curves_xx, etc.) to size N[cite: 21].
- Resizes flat coefficient arrays (alfaijs, gammaijs, etc.) based on the 
  total number of interactions, plus a 25% buffer[cite: 22].
- Resizes thread-local temporary 'A_buffers' based on the maximum 
  number of neighbors, plus a 25% buffer[cite: 24].
"""
function initGIBuffers!(ws::MUSCLWorkspace2D2O, pg::ParticleGrid2D)
    N = length(pg.num_neighbors)

    # 1. Ensure capacity for per-particle buffers (size N)
    ensure_particle_capacity!(ws, N)
    
    # 2. Ensure capacity for flat coefficient buffers (size M + 25%)
    # This existing function already adds the 25% buffer [cite: 22]
    ensure_coefficients_capacity!(ws, pg)
    
    # 3. Find max neighbors for temporary A_buffers
    max_nb = 0
    if !isempty(pg.num_neighbors)
        max_nb = maximum(pg.num_neighbors)
    end
    
    # 4. Ensure capacity for temporary buffers (size max_nb + 25%)
    # This existing function already adds the 25% buffer [cite: 24]
    ensure_capacity!(ws, max_nb)
    
    return nothing
end

# Method for NoLimiter (just copies the slopes)
function limit_slopes!(::NoLimiter, ws::MUSCLWorkspace, grid::ParticleGrid, fVec)
    return
end

# Method for NoLimiter (just copies the slopes)
function limit_slopes!(::NoLimiter, ws::MUSCLWorkspace, grid::ParticleGrid, fVec, f)
    return
end

# Method for classical, ratio-based limiters
function limit_slopes!(strategy::Union{SuperbeeLimiter, MinmodLimiter}, ws::MUSCLWorkspace1D, grid::ParticleGrid1D, fVec)
    for i in 1:grid.N
        val_L, dist_L, val_R, dist_R = find_closest_lr_neighbors_1D(grid, i, fVec)

        slope_L = abs(dist_L) > 1e-12 ?
(fVec[i] - val_L) / (-dist_L) : 0.0
        slope_R = abs(dist_R) > 1e-12 ?
(val_R - fVec[i]) / dist_R : 0.0

        if slope_L * slope_R <= 0.0
            ws.slopes[i] = 0.0
        else
            r = slope_R ≈ 0.0 ?
1.0 : slope_L / slope_R
            phi = strategy isa SuperbeeLimiter ?
superbee_phi(r) : minmod_phi(r)
            ws.slopes[i] = phi * slope_R
        end
    end
end

# Method for geometric, bound-based limiters
function limit_slopes!(strategy::Union{BarthJespersenLimiter, VenkatakrishnanLimiter}, ws::MUSCLWorkspace1D, grid::ParticleGrid1D, fVec)
    for i in 1:grid.N
        ui = fVec[i]
        neighbors = grid.neighbour_indices[i]
        
        if isempty(neighbors) ||
abs(ws.slopes[i]) < 1e-12
            ws.slopes[i] = 0.0
            continue
        end

        u_max_stencil = ui
        u_min_stencil = ui
        for nb_idx in neighbors
            u_max_stencil = max(u_max_stencil, fVec[nb_idx])
            u_min_stencil = min(u_min_stencil, fVec[nb_idx])
   
     end

        phi_i = 1.0
        for nb_idx in neighbors
            dx_ij = getDistance(grid, i, nb_idx)
            delta_recon = ws.slopes[i] * dx_ij 
            
            if abs(delta_recon) < 1e-12;
continue; end
            
            r = if delta_recon > 0.0 # Overshoot
                (u_max_stencil - ui) / delta_recon
            else # Undershoot
                (u_min_stencil - ui) / delta_recon
            end

  
          phi_j = strategy isa BarthJespersenLimiter ?
min(1.0, r) : venkatakrishnan_psi(r)
            phi_i = min(phi_i, phi_j)
        end
        ws.slopes[i] *= clamp(phi_i, 0.0, 1.0)
    end
end

# --- REMOVED old 2D limit_slopes! (logic moved to _limit_slopes helper) ---

function initTimeStep(muscl::MUSCL, particleGrid::ParticleGrid1D, interpAlpha::Real, interpRange::Real)
    ws = muscl.workspace
    N = particleGrid.N
    
    # NEW: Ensure all particle-specific buffers are correctly sized
    ensure_particle_capacity!(ws, N)
    
    for p_idx in 1:particleGrid.N
        neighbors = particleGrid.neighbour_indices[p_idx]
        
        num_neighbors = length(neighbors)
        if num_neighbors == 0;
continue; end

        ensure_capacity!(ws, num_neighbors)
        _ensure_coeff_vectors_sized!(ws, p_idx, num_neighbors)
        
        dxVec = @view ws.dx_buffer[1:num_neighbors]
        wVec  = @view ws.w_buffer[1:num_neighbors]
        A     = @view ws.A_buffer[1:num_neighbors, :]
        
        for (i, nb_idx) in enumerate(neighbors);
dxVec[i] = getDistance(particleGrid, p_idx, nb_idx); end
        muscl.weightFunction(wVec, dxVec; param=interpAlpha, normalisation=particleGrid.dx)
        _compute_muscl_coeffs!(
            muscl.order, dxVec, wVec, A,
            ws.alfaijs[p_idx], ws.alfaij_bars[p_idx], ws.betaijs[p_idx], ws.gammaijs[p_idx]
        )
    end
    calculate_slopes!(muscl.order, muscl.limiter, ws, particleGrid)
end

function calculate_slopes!(order::MUSCLORDER, limiter::AbstractSlopeLimiter, ws::MUSCLWorkspace, grid::ParticleGrid{D}) where {D};
return 
end

# 1D Slope Calculation
function calculate_slopes!(::MUSCLORDER1, limiter::AbstractSlopeLimiter, ws::MUSCLWorkspace1D, grid::ParticleGrid1D)
    fVec = grid.rhos
    for i in 1:grid.N
        ws.slopes[i] = sum(ws.alfaijs[i][k] * (fVec[nb_idx] - fVec[i]) for (k, nb_idx) in enumerate(grid.neighbour_indices[i]))
    end
end

# --- REMOVED old 2D calculate_slopes! (logic moved to _calculate_slopes helper) ---
# --- REMOVED old 2D initGIPos! and initGIRho! ---
# --- REMOVED old 2D _compute_muscl_coeffs! (logic moved to _compute_coeffs! helper) ---


# --- Helper functions for 1D MUSCL orders ---

function _compute_muscl_coeffs!(::MUSCLORDER1, dxVec, wVec, A, alfaij, _, _, _)
    wVec .*= dxVec
    t = dot(wVec, dxVec)
    if abs(t) > 1e-14;
@. alfaij = wVec / t; else; fill!(alfaij, 0.0); end
end

function _compute_muscl_coeffs!(::MUSCLORDER2, dxVec, wVec_orig, A, alfaij, alfaijBar, betaij, _)
    wVec = copy(wVec_orig) # Use a copy to avoid modifying the original weights
    
    # alfa_ij
    wVec .*= dxVec
    t = dot(wVec, dxVec)
    if abs(t) > 1e-14;
@. alfaij = wVec / t; else; fill!(alfaij, 0.0); end

    # alfa_ijBar and betaij
    wVec .= wVec_orig .* dxVec .* dxVec
    A11 = sum(wVec)
    wVec .*= dxVec
    A12 = sum(wVec) / 2
    A22 = dot(wVec, dxVec) / 4
    D = A11 * A22 - A12^2
    
    if abs(D) < 1e-14
        fill!(alfaijBar, 0.0);
fill!(betaij, 0.0)
        return
    end

    wVec .= wVec_orig # Restore original weights
    for i in eachindex(alfaijBar)
        dx_i_sq = dxVec[i]^2
        alfaijBar[i] = (A22 * wVec[i] * dxVec[i] - 0.5 * A12 * wVec[i] * dx_i_sq) / D
        betaij[i] = (0.5 * A11 * wVec[i] * dx_i_sq - A12 * wVec[i] * dxVec[i]) / D
    end
end

function _compute_muscl_coeffs!(::MUSCLORDER3, dxVec, wVec, A, alfaij, alfaijBar, betaij, 
_)
    A_view = @view A[:, 1:3]
    @. A_view[:, 1] = dxVec * wVec 
    @. A_view[:, 2] = (dxVec^2) * wVec / 2
    @. A_view[:, 3] = (dxVec^3) * wVec / 6

    coeff = pinv(A_view; rtol=sqrt(eps(real(float(oneunit(eltype(A)))))))
    @. alfaijBar = coeff[1, :] * wVec
    @. betaij = coeff[2, :] * wVec
    @. alfaij = coeff[3, :] * wVec # Overwrites the order-1 alfaij
end

function _compute_muscl_coeffs!(::MUSCLORDER4, dxVec, wVec, A, alfaij, alfaijBar, betaij, gammaij)
    A_view = @view A[:, 1:4]
    @. A_view[:, 1] = dxVec * wVec 
    @. A_view[:, 2] = (dxVec^2) * wVec / 2
    @. A_view[:, 3] = (dxVec^3) * wVec / 6
    @. A_view[:, 4] = (dxVec^4) * wVec / 24

    coeff = pinv(A_view; rtol=sqrt(eps(real(float(oneunit(eltype(A)))))))
    @. alfaijBar = coeff[1, :] * wVec
    @. betaij = coeff[2, :] * wVec
    @. alfaij = coeff[3, :] * wVec
    @. gammaij = coeff[4, :] * wVec
end

function reconstruct_interface_states(::MUSCLORDER1, particleGrid, fVec, ws, p_idx, nb_idx, deltaPos)
    fij = fVec[p_idx] + 0.5 * deltaPos * ws.slopes[p_idx]
    fji = fVec[nb_idx] - 0.5 * deltaPos * ws.slopes[nb_idx]
    return fij, fji
end

function reconstruct_interface_states(::MUSCLORDER2, particleGrid, fVec, ws, p_idx, nb_idx, deltaPos)
    neighbors_i = particleGrid.neighbour_indices[p_idx]
    neighbors_j = particleGrid.neighbour_indices[nb_idx]
    
    recon_i = sum((deltaPos*ws.alfaij_bars[p_idx][k]/2 + (deltaPos^2)*ws.betaijs[p_idx][k]/8)*(fVec[nb_k] - fVec[p_idx]) for (k, nb_k) in enumerate(neighbors_i))
    recon_j = sum((-deltaPos*ws.alfaij_bars[nb_idx][k]/2 + (deltaPos^2)*ws.betaijs[nb_idx][k]/8)*(fVec[nb_k] - fVec[nb_idx]) for (k, nb_k) in enumerate(neighbors_j))
    
 
   fij = fVec[p_idx] + recon_i
    fji = fVec[nb_idx] + recon_j
    return fij, fji
end

function reconstruct_interface_states(::MUSCLORDER3, particleGrid, fVec, ws, p_idx, nb_idx, deltaPos)
    neighbors_i = particleGrid.neighbour_indices[p_idx]
    neighbors_j = particleGrid.neighbour_indices[nb_idx]
    
    recon_i = sum((deltaPos*ws.alfaij_bars[p_idx][k]/2 + (deltaPos^2)*ws.betaijs[p_idx][k]/8 + (deltaPos^3)*ws.alfaijs[p_idx][k]/48)*(fVec[nb_k] - fVec[p_idx]) for (k, nb_k) in enumerate(neighbors_i))
    recon_j = sum((-deltaPos*ws.alfaij_bars[nb_idx][k]/2 + (deltaPos^2)*ws.betaijs[nb_idx][k]/8 - (deltaPos^3)*ws.alfaijs[nb_idx][k]/48)*(fVec[nb_k] - fVec[nb_idx]) for (k, nb_k) in enumerate(neighbors_j))

    fij = fVec[p_idx] + recon_i
    fji = fVec[nb_idx] + recon_j
    return fij, 
fji
end

function reconstruct_interface_states(::MUSCLORDER4, particleGrid, fVec, ws, p_idx, nb_idx, deltaPos)
    neighbors_i = particleGrid.neighbour_indices[p_idx]
    neighbors_j = particleGrid.neighbour_indices[nb_idx]
    
    recon_i = sum((deltaPos*ws.alfaij_bars[p_idx][k]/2 + (deltaPos^2)*ws.betaijs[p_idx][k]/8 + (deltaPos^3)*ws.alfaijs[p_idx][k]/48 + (deltaPos^4)*ws.gammaijs[p_idx][k]/384)*(fVec[nb_k] - fVec[p_idx]) for (k, nb_k) in enumerate(neighbors_i))
    recon_j = sum((-deltaPos*ws.alfaij_bars[nb_idx][k]/2 + (deltaPos^2)*ws.betaijs[nb_idx][k]/8 - (deltaPos^3)*ws.alfaijs[nb_idx][k]/48 + (deltaPos^4)*ws.gammaijs[nb_idx][k]/384)*(fVec[nb_k] - fVec[nb_idx]) for (k, nb_k) in enumerate(neighbors_j))

    fij = fVec[p_idx] + recon_i
    fji = fVec[nb_idx] + recon_j
    return fij, fji
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


# --- Reconstruction Helpers for fij and fji (2D) ---
# These are kept as they are called by the new functor
# It uses pre-calculated slopes.
function reconstruct_interface_states(::MUSCLORDER1, ws, fi, fj, p_idx, nb_idx, deltaX, deltaY)
    fij = fi  + 0.5 * (deltaX * ws.slopes_x[p_idx]  + deltaY * ws.slopes_y[p_idx])
    fji = fj - 0.5 * (deltaX * ws.slopes_x[nb_idx] + deltaY * ws.slopes_y[nb_idx])
    return fij, fji
end


# --- REFACTORED: Order 2 now just fetches pre-calculated values ---
function reconstruct_interface_states(::MUSCLORDER2, ws, f_i, f_j, p_idx, nb_idx, deltaX, deltaY)
    # Fetch derivatives for particle i
    slope_ix   = ws.slopes_x[p_idx]
    slope_iy   = ws.slopes_y[p_idx]
    curve_xx_i = ws.curves_xx[p_idx]
    curve_yy_i = ws.curves_yy[p_idx]
    curve_xy_i = ws.curves_xy[p_idx]

    # Fetch derivatives for neighbor j
    slope_jx   = ws.slopes_x[nb_idx]
    slope_jy   = ws.slopes_y[nb_idx]
    curve_xx_j = ws.curves_xx[nb_idx]
    curve_yy_j = ws.curves_yy[nb_idx]
    curve_xy_j = ws.curves_xy[nb_idx]

    # Taylor expansion
    h = 0.5 * deltaX
    k = 0.5 * deltaY
    fij = f_i + h*slope_ix + k*slope_iy + 0.5*(h^2*curve_xx_i + 2*h*k*curve_xy_i + k^2*curve_yy_i)
    fji = f_j - h*slope_jx - k*slope_jy + 0.5*(h^2*curve_xx_j + 2*h*k*curve_xy_j + k^2*curve_yy_j)
    
    return fij, fji
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
    f_i::Real,                      # Value of f at particle i
    pg::ParticleGrid,
    f_neighbors::AbstractVector,    # View of neighbor f-values
    df_neighbors::AbstractVector,   # View of neighbor df-values
)::Real where {ORDER<:MUSCLORDER}
    
    div = 0.0
    ws = muscl.workspace
    nFlux = muscl.numericalFlux
    
    if pg.num_neighbors[i] == 0; return 0.0; end

    neighbor_slice = getNBSlice(pg, i)
    dx = pg.neighbor_xdistance
    dy = pg.neighbor_ydistance
    nb_indices = pg.neighbor_indices

    fx, fy = flux(eq, f_i)

    # Loop over neighbors using the local index `k_local`
    @inbounds for k_global in neighbor_slice
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


# --- NEW: Localized Helper Functions for initGI! (2D, Order 1) ---

"""
Calculates 1st-order MUSCL coefficients (alfaij, betaij) for a single particle.
This is the internal logic from the old `_compute_muscl_coeffs!(::MUSCLORDER1, ...)`
"""
@inline function _compute_coeffs!(
    ::MUSCLORDER1,
    nb_slice::UnitRange{Int},
    alfaij::AbstractVector, # View into ws.alfaijs
    betaij::AbstractVector, # View into ws.betaijs
    dx::AbstractVector,            # View of neighbor x-distances
    dy::AbstractVector,            # View of neighbor y-distances
    w::AbstractVector              # View of neighbor weights
)
    # --- 1. Calculate A-matrix components ---
    
   A11 = 0.0; A22 = 0.0; A12 = 0.0
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_k = dx[k]
        dy_k = dy[k]
        A11 += w_k * dx_k * dx_k
        A22 += w_k * dy_k * dy_k
        A12 += w_k * dx_k * dy_k
    end
    
D = A11 * A22 - A12^2

    # --- Handle singular matrix case ---
    if abs(D) < 1e-14
        fill!(alfaij, 0.0)
        fill!(betaij, 0.0)
        return
    end

    # --- 2. Calculate final coefficients ---
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_k = dx[k]
        dy_k = dy[k]

        num_a = w_k * (A22 * dx_k - A12 * dy_k)
        alfaij[k] = num_a / D

        num_b = w_k * (A11 * dy_k - A12 * dx_k)
        betaij[k] = num_b / D
    end
end

"""
Calculates the unlimited slopes for a single particle.
This is the internal logic from the old `calculate_slopes!(::MUSCLORDER1, ...)`
"""
function _calculate_slopes(
    nb_slice::UnitRange{Int64},
    df_neighbors::AbstractVector,   # View of neighbor df-values
    alfaij::AbstractVector, # View of alfaij coefficients
    betaij::AbstractVector  # View of betaij coefficients
)
    slope_x = 0.0
    slope_y = 0.0
    @inbounds for k in nb_slice
        df = df_neighbors[k]
        slope_x += alfaij[k] * df
        slope_y += betaij[k] * df
    end
    return slope_x, slope_y
end

"""
Local slope limiting function (NoLimiter dispatch).
Directly returns the unlimited slopes.
"""
function _limit_slopes(
    limiter::NoLimiter,
    slope_x::Real, slope_y::Real,
    f_i::Real,
    f_neighbors::AbstractVector,
    dx::AbstractVector,
    dy::AbstractVector
)
    return slope_x, slope_y
end

"""
Local slope limiting function (RealSlopeLimiter dispatch).
Applies geometric limiting logic from the old `limit_slopes!(...)`
"""
function _limit_slopes(
    strategy::Union{BarthJespersenLimiter, VenkatakrishnanLimiter},
    slope_x::Real, slope_y::Real,
    f_i::Real,
    f_neighbors::AbstractVector, # View of neighbor f-values
    dx::AbstractVector,          # View of neighbor x-distances
    dy::AbstractVector           # View of neighbor y-distances
)
    ui = f_i
    if slope_x^2 + slope_y^2 < 1e-12
        return 0.0, 0.0
    end

    # Find min/max among neighbors
    u_max = ui
    u_min = ui
    @inbounds for k in eachindex(f_neighbors)
        f_neighbor = f_neighbors[k]
     
        u_max = max(u_max, f_neighbor)
        u_min = min(u_min, f_neighbor)
    end

    phi_i = 1.0
    
    @inbounds for k in eachindex(dx)
     
        delta_recon = slope_x * dx[k] + slope_y * dy[k]
        
        if abs(delta_recon) < 1e-12; continue; end
        
        r = delta_recon > 0.0 ? (u_max - ui) / delta_recon : (u_min - ui) / delta_recon
        phi_j = strategy isa BarthJespersenLimiter ? min(1.0, r) : venkatakrishnan_psi(r)
        phi_i = min(phi_i, phi_j)
    end

    phi_i = clamp(phi_i, 0.0, 1.0)
    
    limited_slope_x = slope_x * phi_i
    limited_slope_y = slope_y * phi_i

    if isnan(limited_slope_y) ||
    isnan(limited_slope_x); error("Found NaN while Limiting!"); end
    
    return limited_slope_x, limited_slope_y
end


# --- 3. NEW: MAIN LOCALIZED FUNCTION FOR SLOPE CALCULATION (2D, Order 1) ---

"""
    initGI!(muscl, i, f_i, pg, neighbor_fs, neighbor_dfs)

Rewritten initGI! to be allocation-friendly.

This function takes the top-level grid and flat data buffers.
It is responsible for creating all the necessary views internally,
which allows the compiler to optimize away the allocations.
"""
function initGI!(
    muscl::MUSCL{2,MUSCLORDER1},
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    pg::ParticleGrid,               # The entire grid object
    neighbor_fs::AbstractVector,    # The flat neighbor-value buffer
    neighbor_dfs::AbstractVector    # The flat neighbor-difference buffer
)
    ws = muscl.workspace
    
    # --- 1. Get particle-specific indices ---
    num_nb = pg.num_neighbors[i]
    if num_nb == 0
        ws.slopes_x[i] = 0.0
        ws.slopes_y[i] = 0.0
        return
    end
    
    neighbor_slice = getNBSlice(pg, i)
    

    dx = pg.neighbor_xdistance
    dy = pg.neighbor_ydistance
    w  = pg.neighbor_weights
    # 'neighbors' (indices) isn't needed by these helpers, so we skip it.

    # --- 3. Call the (already clean) helpers ---
    _compute_coeffs!(muscl.order, neighbor_slice, ws.alfaijs, ws.betaijs, dx, dy, w)
    
    slope_x, slope_y = _calculate_slopes(neighbor_slice, neighbor_dfs, ws.alfaijs, ws.betaijs)

    # Note: _limit_slopes was not provided, but assuming it has the same signature
    slope_x, slope_y = _limit_slopes(muscl.limiter, slope_x, slope_y, f_i, neighbor_fs, dx, dy)
    
    if isnan(slope_y) || isnan(slope_x)
        error("Found NaN while calculating slopes for a particle!")
    end

    # Store the final, limited slopes
    ws.slopes_x[i] = slope_x
    ws.slopes_y[i] = slope_y
    return
end

"""
    initGI!(muscl, thread_idx, i, f_i, num_nb, neighbor_slice, ...)

Main function to calculate and store 2nd-order derivatives (slopes and curves)
for a *single* particle `i`. This is intended to be called inside a parallel loop.
"""
function initGI!(
    muscl::MUSCL{2,MUSCLORDER2},
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    pg::ParticleGrid, 
    f_neighbors::AbstractVector,    # View of neighbor f-values
    df_neighbors::AbstractVector,   # View of neighbor df-values
)
    ws = muscl.workspace
    thread_idx = Threads.threadid() - 1
    # --- Fetch the correct thread-local buffer ---
    A_buffer_threadlocal = ws.A_buffers[thread_idx] # <--- This now works

    # Get views into the correct slice of the global coefficient buffers
    alfaij      = ws.alfaijs
    betaij      = ws.betaijs
    alfaij_bar  = ws.alfaij_bars
    betaij_bar  = ws.betaij_bars
    gammaij     = ws.gammaijs
    dx          = pg.neighbor_xdistance
    dy          = pg.neighbor_ydistance
    w          = pg.neighbor_weights
    num_nb      = pg.num_neighbors[i]

    nb_slice = getNBSlice(pg, i)

    # --- Handle zero-neighbor case ---
    if num_nb == 0
        for k = nb_slice
            alfaij[k] = 0.
            betaij[k] = 0.
            alfaij_bar[k] = 0.
            betaij_bar[k] = 0.
            gammaij[k] = 0.
        end
        ws.slopes_x[i] = 0.0
        ws.slopes_y[i] = 0.0
        ws.curves_xx[i] = 0.0
        ws.curves_yy[i] = 0.0
        ws.curves_xy[i] = 0.0
        return
    end

    # --- 1. Compute Coefficients ---
    A_view = @view A_buffer_threadlocal[1:num_nb, 1:5]
    
    _compute_coeffs!(
        muscl.order, nb_slice,
        alfaij, betaij, alfaij_bar,
        betaij_bar, gammaij,
        A_view, dx, dy, w
    )
    
    # --- 2. Calculate Derivatives ---
    # <--- 3. PASS df_neighbors, NOT f_i and f_neighbors ---
    (slope_x, slope_y, curve_xx, curve_yy, curve_xy) = _calculate_derivatives(
        muscl.order, nb_slice, df_neighbors,
        alfaij, betaij, alfaij_bar,
        betaij_bar, gammaij
    )

    # --- 3. Limit Derivatives ---
    (slope_x, slope_y, curve_xx, curve_yy, curve_xy) = _limit_derivatives(
        muscl.order, nb_slice, muscl.limiter,
        slope_x, slope_y, curve_xx, curve_yy, curve_xy,
        f_i, f_neighbors, dx, dy
    )

    # --- 4. Store Final Derivatives ---
    if isnan(slope_y) || isnan(slope_x) || isnan(curve_xx)
        error("Found NaN while calculating derivatives for a particle!")
    end

    ws.slopes_x[i]  = slope_x
    ws.slopes_y[i]  = slope_y
    ws.curves_xx[i] = curve_xx
    ws.curves_yy[i] = curve_yy
    ws.curves_xy[i] = curve_xy
    return
end


"""
Calculates 2nd-order MUSCL coefficients for a single particle.
This is the internal logic from the old `_compute_muscl_coeffs!(::MUSCLORDER2, ...)`
"""
function _compute_coeffs!(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    alfaij::AbstractVector,     # View into ws.alfaijs
    betaij::AbstractVector,     # View into ws.betaijs
    alfaij_bar::AbstractVector, # View into ws.alfaij_bars
    betaij_bar::AbstractVector, # View into ws.betaij_bars
    gammaij::AbstractVector,    # View into ws.gammaijs
    A_view::AbstractMatrix,            # View of thread-local temp matrix
    dx::AbstractVector,                # View of neighbor x-distances
    dy::AbstractVector,                # View of neighbor y-distances
    w::AbstractVector                  # View of neighbor weights
)
    count = 0
    # --- 1. Populate the temporary A matrix ---
    @inbounds for k in nb_slice
        count += 1

        dx_k = dx[k]
        dy_k = dy[k]
        w_k  = w[k]

        A_view[count, 1] = dx_k * w_k
        A_view[count, 2] = dy_k * w_k
        A_view[count, 3] = (dx_k^2) * w_k * 0.5
        A_view[count, 4] = (dy_k^2) * w_k * 0.5
        A_view[count, 5] = dx_k * dy_k * w_k
    end

    # --- 2. Perform the pseudo-inverse (this is a necessary allocation) ---
    coeff = pinv(A_view, rtol=sqrt(eps(real(float(oneunit(eltype(A_view)))))))
    
    # --- 3. Write final coefficients ---
    count = 0
    @inbounds for k in nb_slice
        count += 1
        w_k = w[k]
        alfaij[k]     = coeff[1, count] * w_k
        betaij[k]     = coeff[2, count] * w_k
        alfaij_bar[k] = coeff[3, count] * w_k
        betaij_bar[k] = coeff[4, count] * w_k
        gammaij[k]    = coeff[5, count] * w_k
    end
end

"""
Calculates the unlimited 2nd-order derivatives for a single particle.
This version uses the pre-calculated neighbor_dfs buffer.
"""
function _calculate_derivatives(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    df_neighbors::AbstractVector, # <--- CHANGED
    alfaij::AbstractVector,
    betaij::AbstractVector,
    alfaij_bar::AbstractVector,
    betaij_bar::AbstractVector,
    gammaij::AbstractVector
)
    slope_x = 0.0
    slope_y = 0.0
    curve_xx = 0.0
    curve_yy = 0.0
    curve_xy = 0.0
    
    @inbounds for k in nb_slice
        df = df_neighbors[k] # <--- CHANGED
        
        slope_x  += alfaij[k] * df
        slope_y  += betaij[k] * df
        curve_xx += alfaij_bar[k] * df
        curve_yy += betaij_bar[k] * df
        curve_xy += gammaij[k] * df
    end

    return slope_x, slope_y, curve_xx, curve_yy, curve_xy
end

"""
Local derivative limiting function (NoLimiter dispatch for MUSCLORDER2).
"""
function _limit_derivatives(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int64},
    limiter::NoLimiter,
    slope_x, slope_y, curve_xx, curve_yy, curve_xy,
    f_i, f_neighbors, dx, dy
)
    # No limiting, just pass through
    return slope_x, slope_y, curve_xx, curve_yy, curve_xy
end

"""
Local derivative limiting function (RealSlopeLimiter dispatch for MUSCLORDER2).
Based on the assertion, this is not implemented.
"""
function _limit_derivatives(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    limiter::RealSlopeLimiter,
    slope_x, slope_y, curve_xx, curve_yy, curve_xy,
    f_i, f_neighbors, dx, dy
)
    # Limiting is only implemented for 1st order.
    # We must set 2nd-order derivatives to zero if we limit 1st-order ones,
    # but since limiting logic isn't defined, we just error out.
    error("Slope limiting is not implemented for MUSCLORDER2.")
    
    # Or, if we wanted to just limit the slopes and kill curves:
    # limited_sx, limited_sy = _limit_slopes(limiter, slope_x, slope_y, f_i, f_neighbors, dx, dy)
    # return limited_sx, limited_sy, 0.0, 0.0, 0.0
end