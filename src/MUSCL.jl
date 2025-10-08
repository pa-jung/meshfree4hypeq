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
  closest left and right neighbors. Returns `0.0` for values and distances if a 
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

    # --- Temporary buffers for neighbor-specific calculations ---
    dx_buffer::Vector{Float64}
    w_buffer::Vector{Float64}
    A_buffer::Matrix{Float64}
    
    max_neighbors::Int

    function MUSCLWorkspace1D(N_particles::Int = 1, initial_capacity::Int=20)
        new(
            [Float64[] for _ in 1:N_particles], [Float64[] for _ in 1:N_particles],
            [Float64[] for _ in 1:N_particles], [Float64[] for _ in 1:N_particles],
            Vector{Float64}(undef,N_particles),
            zeros(Float64, initial_capacity), zeros(Float64, initial_capacity),
            zeros(Float64, initial_capacity, 4), initial_capacity
        )
    end
end

mutable struct MUSCLWorkspace2D <: MUSCLWorkspace
    # Per-particle coefficient storage (ragged arrays)
    alfaijs::Vector{Vector{Float64}}; betaijs::Vector{Vector{Float64}}
    alfaij_bars::Vector{Vector{Float64}}; betaij_bars::Vector{Vector{Float64}}
    gammaijs::Vector{Vector{Float64}}
    
    slopes_x::Vector{Float64}; slopes_y::Vector{Float64} # Separate vectors for 2D slopes
    
    # Temporary buffers
    dx_buffer::Vector{Float64}; dy_buffer::Vector{Float64}; w_buffer::Vector{Float64}
    A_buffer::Matrix{Float64}
    max_neighbors::Int

    function MUSCLWorkspace2D(N_particles::Int=160, cap=40)
        new([Float64[] for _ in 1:N_particles], [Float64[] for _ in 1:N_particles],
            [Float64[] for _ in 1:N_particles], [Float64[] for _ in 1:N_particles],
            [Float64[] for _ in 1:N_particles],
            zeros(N_particles), zeros(N_particles),
            zeros(cap), zeros(cap), zeros(cap), zeros(cap, 5), cap)
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

function ensure_capacity!(ws::MUSCLWorkspace2D, n::Int)
    if n > length(ws.dx_buffer)
        new_capacity = n + n ÷ 4
        
        resize!.((ws.dx_buffer, ws.w_buffer, ws.dy_buffer), new_capacity)
        
        # Preserve the number of columns from the old matrix
        num_cols = size(ws.A_buffer, 2)
        ws.A_buffer = Matrix{Float64}(undef, new_capacity, num_cols)
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

function _ensure_coeff_vectors_sized!(ws::MUSCLWorkspace2D, p_idx::Int, n::Int)
    if length(ws.alfaijs[p_idx]) < n
        new_capacity = n #+ n ÷ 4
        
        resize!.((
            ws.alfaijs[p_idx], 
            ws.alfaij_bars[p_idx], 
            ws.betaijs[p_idx], 
            ws.betaij_bars[p_idx], # 2D specific
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
    end
    return nothing
end

function ensure_particle_capacity!(ws::MUSCLWorkspace2D, N::Int)
    current_size = length(ws.alfaijs)
    if current_size < N
        new_capacity = N #+ N ÷ 4
        num_to_add = new_capacity - current_size
        
        for _ in 1:num_to_add
            push!(ws.alfaijs, Float64[])
            push!(ws.alfaij_bars, Float64[])
            push!(ws.betaijs, Float64[])
            push!(ws.betaij_bars, Float64[]) # 2D specific
            push!(ws.gammaijs, Float64[])
        end
        println(size(ws.alfaijs))
        resize!.((ws.slopes_x, ws.slopes_y), new_capacity)
    end
    return nothing
end

struct MUSCL{D,ORDER<:MUSCLORDER, L<:AbstractSlopeLimiter, WS<:MUSCLWorkspace} <: GradientInterpolator
    order::ORDER
    limiter::L
    res::Vector{Float64}
    weightFunction::MLSWeightFunction
    numericalFlux::NumericalFluxFunction
    workspace::WS
    
end
function MUSCL(
    order::Int, 
    dimension::Int; 
    limiter::L=NoLimiter(), 
    weightFunction=exponentialWeightFunction(), 
    numericalFlux=RusanovFlux()
) where {L<:AbstractSlopeLimiter}
    
    # Your assertion was checking for order==2 but the message said order 1, I've corrected it.
    if !(limiter isa NoLimiter)
        @assert order == 1 "Slope limiting is currently only implemented for MUSCL order 1."
    end

    ws = dimension == 1 ? MUSCLWorkspace1D() : MUSCLWorkspace2D()
    res_size = (dimension == 1) ? order : (order == 1 ? 2 : 5) # Determine size of result buffer
    
    WS = typeof(ws)
    # Call the simple default constructor with the correct, inferred types.
    if order == 1
        return MUSCL{dimension, MUSCLORDER1, L, WS}(MUSCLORDER1(), limiter, zeros(res_size), weightFunction, numericalFlux, ws)
    elseif order == 2
        return MUSCL{dimension, MUSCLORDER2, L, WS}(MUSCLORDER2(), limiter, zeros(res_size), weightFunction, numericalFlux, ws)
    elseif order == 3
        return MUSCL{dimension, MUSCLORDER3, L, WS}(MUSCLORDER3(), limiter, zeros(res_size), weightFunction, numericalFlux, ws)
    elseif order == 4
        return MUSCL{dimension, MUSCLORDER4, L, WS}(MUSCLORDER4(), limiter, zeros(res_size), weightFunction, numericalFlux, ws)
    else
        error("Order must be 1, 2, 3, or 4.")
    end
end
# Method for NoLimiter (just copies the slopes)
function limit_slopes!(::NoLimiter, ws::MUSCLWorkspace, grid::ParticleGrid, fVec)
    return
end

# Method for classical, ratio-based limiters
function limit_slopes!(strategy::Union{SuperbeeLimiter, MinmodLimiter}, ws::MUSCLWorkspace1D, grid::ParticleGrid1D, fVec)
    for i in 1:grid.N
        val_L, dist_L, val_R, dist_R = find_closest_lr_neighbors_1D(grid, i, fVec)

        slope_L = abs(dist_L) > 1e-12 ? (fVec[i] - val_L) / (-dist_L) : 0.0
        slope_R = abs(dist_R) > 1e-12 ? (val_R - fVec[i]) / dist_R : 0.0

        if slope_L * slope_R <= 0.0
            ws.slopes[i] = 0.0
        else
            r = slope_R ≈ 0.0 ? 1.0 : slope_L / slope_R
            phi = strategy isa SuperbeeLimiter ? superbee_phi(r) : minmod_phi(r)
            ws.slopes[i] = phi * slope_R
        end
    end
end

# Method for geometric, bound-based limiters
function limit_slopes!(strategy::Union{BarthJespersenLimiter, VenkatakrishnanLimiter}, ws::MUSCLWorkspace1D, grid::ParticleGrid1D, fVec)
    for i in 1:grid.N
        ui = fVec[i]
        neighbors = grid.neighbour_indices[i]
        
        if isempty(neighbors) || abs(ws.slopes[i]) < 1e-12
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
            
            if abs(delta_recon) < 1e-12; continue; end
            
            r = if delta_recon > 0.0 # Overshoot
                (u_max_stencil - ui) / delta_recon
            else # Undershoot
                (u_min_stencil - ui) / delta_recon
            end

            phi_j = strategy isa BarthJespersenLimiter ? min(1.0, r) : venkatakrishnan_psi(r)
            phi_i = min(phi_i, phi_j)
        end
        ws.slopes[i] *= clamp(phi_i, 0.0, 1.0)
    end
end

# 2D geometric limiter (in-place)
function limit_slopes!(strategy::Union{BarthJespersenLimiter, VenkatakrishnanLimiter}, ws::MUSCLWorkspace2D, grid::ParticleGrid2D, fVec)
    
    for i in 1:grid.N
        ui = fVec[i]
        neighbors = grid.neighbour_indices[i]
        sigma_i_unlimited = (ws.slopes_x[i], ws.slopes_y[i])
        
        if isempty(neighbors) || norm(sigma_i_unlimited) < 1e-12; ws.slopes_x[i]=0.0; ws.slopes_y[i]=0.0; continue; end

        u_max, u_min = ui, ui
        for nb_idx in neighbors; u_max=max(u_max, fVec[nb_idx]); u_min=min(u_min, fVec[nb_idx]); end

        phi_i = 1.0
        for nb_idx in neighbors
            dx_ij = getDistance(grid, i, nb_idx)
            delta_recon = dot(sigma_i_unlimited, dx_ij)
            if abs(delta_recon) < 1e-12; continue; end
            
            r = delta_recon > 0.0 ? (u_max - ui)/delta_recon : (u_min - ui)/delta_recon
            phi_j = strategy isa BarthJespersenLimiter ? min(1.0, r) : venkatakrishnan_psi(r)
            phi_i = min(phi_i, phi_j)
        end
        phi_i = clamp(phi_i, 0.0, 1.0)
        ws.slopes_x[i] *= phi_i
        ws.slopes_y[i] *= phi_i
        if isnan(ws.slopes_y[i]) || isnan(ws.slopes_x[i]); error("Found NaN while Limiting!") end
    end
end

function initTimeStep(muscl::MUSCL, particleGrid::ParticleGrid1D, interpAlpha::Real, interpRange::Real)
    ws = muscl.workspace
    N = particleGrid.N
    
    # NEW: Ensure all particle-specific buffers are correctly sized
    ensure_particle_capacity!(ws, N)
    
    for p_idx in 1:particleGrid.N
        neighbors = particleGrid.neighbour_indices[p_idx]
        
        num_neighbors = length(neighbors)
        if num_neighbors == 0; continue; end

        ensure_capacity!(ws, num_neighbors)
        _ensure_coeff_vectors_sized!(ws, p_idx, num_neighbors)
        
        dxVec = @view ws.dx_buffer[1:num_neighbors]
        wVec  = @view ws.w_buffer[1:num_neighbors]
        A     = @view ws.A_buffer[1:num_neighbors, :]
        
        for (i, nb_idx) in enumerate(neighbors); dxVec[i] = getDistance(particleGrid, p_idx, nb_idx); end
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
    limit_slopes!(limiter, ws, grid, fVec)   
end

# 2D Slope Calculation
function calculate_slopes!(::MUSCLORDER1, limiter::AbstractSlopeLimiter, ws::MUSCLWorkspace2D, grid::ParticleGrid2D)
    fVec = grid.rhos
    for i in 1:grid.N
        ws.slopes_x[i] = sum(ws.alfaijs[i][k] * (fVec[nb_idx] - fVec[i]) for (k, nb_idx) in enumerate(grid.neighbour_indices[i]))
        ws.slopes_y[i] = sum(ws.betaijs[i][k] * (fVec[nb_idx] - fVec[i]) for (k, nb_idx) in enumerate(grid.neighbour_indices[i]))
    end
    if any(isnan(s) for s = ws.slopes_x)
        error("Found NaN before limiting!")
    end
    limit_slopes!(limiter, ws, grid, fVec)
end

# --- Helper functions for each MUSCL order ---

function _compute_muscl_coeffs!(::MUSCLORDER1, dxVec, wVec, A, alfaij, _, _, _)
    wVec .*= dxVec
    t = dot(wVec, dxVec)
    if abs(t) > 1e-14; @. alfaij = wVec / t; else; fill!(alfaij, 0.0); end
end

function _compute_muscl_coeffs!(::MUSCLORDER2, dxVec, wVec_orig, A, alfaij, alfaijBar, betaij, _)
    wVec = copy(wVec_orig) # Use a copy to avoid modifying the original weights
    
    # alfa_ij
    wVec .*= dxVec
    t = dot(wVec, dxVec)
    if abs(t) > 1e-14; @. alfaij = wVec / t; else; fill!(alfaij, 0.0); end

    # alfa_ijBar and betaij
    wVec .= wVec_orig .* dxVec .* dxVec
    A11 = sum(wVec)
    wVec .*= dxVec
    A12 = sum(wVec) / 2
    A22 = dot(wVec, dxVec) / 4
    D = A11 * A22 - A12^2
    
    if abs(D) < 1e-14
        fill!(alfaijBar, 0.0); fill!(betaij, 0.0)
        return
    end

    wVec .= wVec_orig # Restore original weights
    for i in eachindex(alfaijBar)
        dx_i_sq = dxVec[i]^2
        alfaijBar[i] = (A22 * wVec[i] * dxVec[i] - 0.5 * A12 * wVec[i] * dx_i_sq) / D
        betaij[i] = (0.5 * A11 * wVec[i] * dx_i_sq - A12 * wVec[i] * dxVec[i]) / D
    end
end

function _compute_muscl_coeffs!(::MUSCLORDER3, dxVec, wVec, A, alfaij, alfaijBar, betaij, _)
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
    return fij, fji
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

# function (muscl::MUSCL{1, MUSCLORDER1,L})(
#     particleGrid::ParticleGrid1D, 
#     particleIndex::Integer, 
#     fVec::AbstractVector, 
#     eq::ScalarHyperbolicPDE, 
#     settings::SimSetting; 
#     setCurvature::Bool=true
# ) where {L <: RealSlopeLimiter}
#     div = 0.0
#     ws = muscl.workspace
    
#     # Retrieve the pre-calculated LIMITED slope for the current particle
#     sigma_i_lim = ws.slopes[particleIndex]

#     for (index_in_list, nbIndex) in enumerate(particleGrid.neighbour_indices[particleIndex])
#         deltaPos = getDistance(particleGrid, particleIndex, nbIndex)
        
#         # Retrieve the pre-calculated LIMITED slope for the neighbor
#         sigma_j_lim = ws.slopes[nbIndex]

#         # Reconstruct states at midpoint using LIMITED slopes
#         fij = fVec[particleIndex] + 0.5 * deltaPos * sigma_i_lim
#         fji = fVec[nbIndex] - 0.5 * deltaPos * sigma_j_lim
        
#         fm, fp = sortFlux(fij, fji, deltaPos)
#         num_flux = muscl.numericalFlux(fm, fp, eq)
        
#         # Divergence sum uses the geometric alfaij coefficients
#         div += ws.alfaijs[particleIndex][index_in_list] * (num_flux - flux(eq, fVec[particleIndex]))
#     end

#     if setCurvature; particleGrid.curvatures[particleIndex] = 0.0; end
    
#     return 2.0 * div
# end

# --- 4. Refactored MUSCL Functor ---
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

# The MUSCL struct itself doesn't need to change, but its constructor
# will now create the correct 1D or 2D workspace based on the grid info.
# This requires adding N_particles to the 2D constructor as well.


# --- 2. Refactored `initTimeStep` for 2D MUSCL ---

function initTimeStep(muscl::MUSCL, particleGrid::ParticleGrid2D, interpAlpha::Real, interpRange::Real)
    ws = muscl.workspace # Assumes this is a MUSCLWorkspace2D
    N = particleGrid.N
    ensure_particle_capacity!(ws, N)

    for p_idx in 1:N
        neighbors = particleGrid.neighbour_indices[p_idx]
        num_neighbors = length(neighbors)
        if num_neighbors == 0; continue; end

        # Efficiently resize buffers and coefficient vectors
        ensure_capacity!(ws, num_neighbors) # Needs to be adapted for 2D buffers
        _ensure_coeff_vectors_sized!(ws, p_idx, num_neighbors)

        dxVec = @view ws.dx_buffer[1:num_neighbors]
        dyVec = @view ws.dy_buffer[1:num_neighbors]
        wVec  = @view ws.w_buffer[1:num_neighbors]
        A     = @view ws.A_buffer[1:num_neighbors, :]
        
        for (i, nb_idx) in enumerate(neighbors)
            dx, dy = getDistance(particleGrid, p_idx, nb_idx)
            dxVec[i], dyVec[i] = dx, dy
        end
        muscl.weightFunction(wVec, dxVec, dyVec; param=interpAlpha, normalisation=interpRange)

        _compute_muscl_coeffs!(
            muscl.order, dxVec, dyVec, wVec, A,
            ws.alfaijs[p_idx], ws.betaijs[p_idx], ws.alfaij_bars[p_idx], 
            ws.betaij_bars[p_idx], ws.gammaijs[p_idx]
        )
    end
    calculate_slopes!(muscl.order, muscl.limiter, ws, particleGrid)
    
end

# --- Coefficient Calculation & Sizing Helpers for 2D ---
function _compute_muscl_coeffs!(::MUSCLORDER1, dxVec, dyVec, wVec, A, alfaij, betaij, _, _, _)
    # --- 1. Calculate coefficients without allocations and store in the buffer `A` ---
    # The sum-generator pattern is still the key to avoiding temporary arrays here.
    A[1, 1] = sum(wVec[i] * dxVec[i]^2 for i in eachindex(wVec, dxVec))      # A11
    A[2, 2] = sum(wVec[i] * dyVec[i]^2 for i in eachindex(wVec, dyVec))      # A22
    A[1, 2] = sum(wVec[i] * dxVec[i] * dyVec[i] for i in eachindex(wVec, dxVec, dyVec)) # A12
    # The matrix is symmetric, so A[2, 1] can be a copy
    A[2, 1] = A[1, 2]

    # --- 2. Calculate the determinant from the buffer ---
    D = A[1, 1] * A[2, 2] - A[1, 2]^2
    
    if abs(D) < 1e-14
        fill!(alfaij, 0.0)
        fill!(betaij, 0.0)
        return
    end

    # --- 3. Compute final vectors using the buffered values ---
    # This fully broadcasted syntax still ensures a single, fused, zero-allocation loop.
    A11 = A[1, 1]
    A22 = A[2, 2]
    A12 = A[1, 2]
    alfaij .= (wVec .* (A22 .* dxVec .- A12 .* dyVec)) ./ D
    betaij .= (wVec .* (A11 .* dyVec .- A12 .* dxVec)) ./ D
end

function _compute_muscl_coeffs!(::MUSCLORDER2, dxVec, dyVec, wVec, A, alfaij, betaij, alfaijBar, betaijBar, gammaij)
    A_view = @view A[:, 1:5]
    @. A_view[:, 1] = dxVec * wVec
    @. A_view[:, 2] = dyVec * wVec
    @. A_view[:, 3] = (dxVec^2) * wVec / 2
    @. A_view[:, 4] = (dyVec^2) * wVec / 2
    @. A_view[:, 5] = dxVec * dyVec * wVec

    # The critical fix you found for numerical stability
    coeff = pinv(A_view, rtol=sqrt(eps(real(float(oneunit(eltype(A)))))))
    
    @. alfaij = coeff[1, :] * wVec
    @. betaij = coeff[2, :] * wVec
    @. alfaijBar = coeff[3, :] * wVec
    @. betaijBar = coeff[4, :] * wVec
    @. gammaij = coeff[5, :] * wVec
end


# --- Reconstruction Helpers for fij and fji (2D) ---
function reconstruct_interface_states(::MUSCLORDER1, particleGrid::ParticleGrid2D, fVec, ws, p_idx, nb_idx, deltaX, deltaY)
    if isnan(ws.slopes_x[p_idx]) || isnan(ws.slopes_x[nb_idx]) || isnan(ws.slopes_y[p_idx]) || isnan(ws.slopes_y[nb_idx]); 
        println(p_idx,nb_idx)
        println(isnan(ws.slopes_x[p_idx]) || isnan(ws.slopes_x[nb_idx]) || isnan(ws.slopes_y[p_idx]) || isnan(ws.slopes_y[nb_idx]))
        error("NaN while reconstructing from slopes!")
    end
    #println(ws.slopes_y[setdiff(1:N, particleGrid.interior_indices)])
    #if !(nb_idx in particleGrid.interior_indices) && fVec[nb_idx] != 0.; println("ghost",fVec[nb_idx],"particle",fVec[p_idx],"Index:",(nb_idx,p_idx)) end
    fij = fVec[p_idx] + 0.5 * (deltaX * ws.slopes_x[p_idx] + deltaY * ws.slopes_y[p_idx])
    fji = fVec[nb_idx] - 0.5 * (deltaX * ws.slopes_x[nb_idx] + deltaY * ws.slopes_y[nb_idx])
    if isnan(fij) || isnan(fji); error("Found NaN while reconstructing! $fVec") end
    return fij, fji
end
# function reconstruct_interface_states(::MUSCLORDER1, particleGrid::ParticleGrid2D, fVec, ws, p_idx, nb_idx, deltaX, deltaY)
#     f_i = fVec[p_idx]
#     f_j = fVec[nb_idx]
    
#     # Calculate derivatives at particle i (once)
#     slope_ix = sum(ws.alfaijs[p_idx][k] * (fVec[nb_k] - f_i) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[p_idx]))
#     slope_iy = sum(ws.betaijs[p_idx][k] * (fVec[nb_k] - f_i) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[p_idx]))
    
#     # Calculate derivatives at neighbor j (once)
#     slope_jx = sum(ws.alfaijs[nb_idx][k] * (fVec[nb_k] - f_j) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[nb_idx]))
#     slope_jy = sum(ws.betaijs[nb_idx][k] * (fVec[nb_k] - f_j) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[nb_idx]))

#     # Apply Taylor expansion for interface values
#     fij = f_i + 0.5 * (deltaX * slope_ix + deltaY * slope_iy)
#     fji = f_j - 0.5 * (deltaX * slope_jx + deltaY * slope_jy)
#     return fij, fji
# end

function reconstruct_interface_states(::MUSCLORDER2, particleGrid::ParticleGrid2D, fVec, ws, p_idx, nb_idx, deltaX, deltaY)
    f_i = fVec[p_idx]
    f_j = fVec[nb_idx]
    
    # Calculate derivatives at particle i (once)
    slope_ix = sum(ws.alfaijs[p_idx][k] * (fVec[nb_k] - f_i) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[p_idx]))
    slope_iy = sum(ws.betaijs[p_idx][k] * (fVec[nb_k] - f_i) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[p_idx]))
    curve_xx = sum(ws.alfaij_bars[p_idx][k] * (fVec[nb_k] - f_i) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[p_idx]))
    curve_yy = sum(ws.betaij_bars[p_idx][k] * (fVec[nb_k] - f_i) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[p_idx]))
    curve_xy = sum(ws.gammaijs[p_idx][k] * (fVec[nb_k] - f_i) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[p_idx]))

    # Calculate derivatives at neighbor j (once)
    slope_jx = sum(ws.alfaijs[nb_idx][k] * (fVec[nb_k] - f_j) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[nb_idx]))
    slope_jy = sum(ws.betaijs[nb_idx][k] * (fVec[nb_k] - f_j) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[nb_idx]))
    curve_xx_j = sum(ws.alfaij_bars[nb_idx][k] * (fVec[nb_k] - f_j) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[nb_idx]))
    curve_yy_j = sum(ws.betaij_bars[nb_idx][k] * (fVec[nb_k] - f_j) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[nb_idx]))
    curve_xy_j = sum(ws.gammaijs[nb_idx][k] * (fVec[nb_k] - f_j) for (k, nb_k) in enumerate(particleGrid.neighbour_indices[nb_idx]))

    h = 0.5 * deltaX
    k = 0.5 * deltaY
    fij = f_i + h*slope_ix + k*slope_iy + 0.5*(h^2*curve_xx + 2*h*k*curve_xy + k^2*curve_yy)
    fji = f_j - h*slope_jx - k*slope_jy + 0.5*(h^2*curve_xx_j + 2*h*k*curve_xy_j + k^2*curve_yy_j)
    
    return fij, fji
end


# --- 2. Refactored MUSCL Functor for 2D (Corrected Logic) ---
function (muscl::MUSCL{2, ORDER})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector, 
    eq::ScalarHyperbolicPDE, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real where {ORDER<:MUSCLORDER}
    
    div = 0.0
    ws = muscl.workspace::MUSCLWorkspace2D
    
    alfaij_i = ws.alfaijs[particleIndex]
    betaij_i = ws.betaijs[particleIndex]

    for (index_in_list, nbIndex) in enumerate(particleGrid.neighbour_indices[particleIndex])
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        
        # Call the correct, non-looping reconstruction helper
        fij, fji = reconstruct_interface_states(muscl.order, particleGrid, fVec, ws, particleIndex, nbIndex, deltaX, deltaY)

        fmx, fpx, fmy, fpy = sortFlux(fij, fji, deltaX, deltaY)
        fx, fy = flux(eq, fVec[particleIndex])
        
        # The divergence sum is now correct
        div += alfaij_i[index_in_list] * (muscl.numericalFlux(fmx, fpx, eq, 1) - fx) + 
               betaij_i[index_in_list] * (muscl.numericalFlux(fmy, fpy, eq, 2) - fy)
    end

    if setCurvature
        _set_curvature!(muscl.order, particleGrid, fVec, ws, particleIndex)
    end
    
    return 2 * div
end


# --- Curvature Helper for 2D ---
function _set_curvature!(::MUSCLORDER1, grid, fVec, ws, p_idx)
    grid.curvatures[p_idx, :] .= 0.0
end

function _set_curvature!(::MUSCLORDER2, grid::ParticleGrid2D, fVec, ws, p_idx)
    neighbors = grid.neighbour_indices[p_idx]
    f_i = fVec[p_idx]
    
    # Retrieve pre-computed coefficients from the workspace
    alfaijBar_i = ws.alfaij_bars[p_idx]
    betaijBar_i = ws.betaij_bars[p_idx]
    
    c_xx = sum(alfaijBar_i[k] * (fVec[nb_k] - f_i) for (k, nb_k) in enumerate(neighbors))
    c_yy = sum(betaijBar_i[k] * (fVec[nb_k] - f_i) for (k, nb_k) in enumerate(neighbors))
    
    grid.curvatures[p_idx, 1] = c_xx
    grid.curvatures[p_idx, 2] = c_yy
end