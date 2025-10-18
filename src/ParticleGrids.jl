module ParticleGrids

export ParticleGrid, ParticleGrid1D, ParticleGrid2D, getPeriodicDistance, saveGrid, plotDensity, 
       animateDensity, getTimeStep, findLocalExtrema!, updateVoxelInformation!, gridToLinearIndex, linearIndexToGrid, 
       findneighboringVoxels, updateNeighbors!, getEuclideanDistance, logMOODEvents!, findLocalExtremaAbs!, 
       determineVolumes!, getDistance, apply_boundary_conditions!, ParticleGridSystem, set_df!

using FileIO, JLD2
using Plots
using Printf
using LaTeXStrings
using Statistics
using LinearAlgebra
using CellListMap
using StaticArrays
using ..SimSettings
using ..HyperbolicPDEs
using ..MLSWeightFunctions
import Meshfree4ScalarEq

# # --- Export new types and functions ---
# export ParticleGrid, ParticleGrid1D, ParticleGrid2D, setInitialConditions!, 
#        getDistance, saveGrid, plotDensity, animateDensity, getTimeStep, 
#        findLocalExtrema!, updateNeighbors!, determineVolumes!, 
#        apply_boundary_conditions!, ParticleGridSystem

# --- Core Abstract Type and System Alias ---
abstract type ParticleGrid{D} end # Now parameterized by dimension
const ParticleGridSystem{N, D} = NTuple{N, <:ParticleGrid{D}}


#==============================================================================
  1D PARTICLE GRID (Struct of Arrays Implementation)
==============================================================================#

struct ParticleGrid1D <: ParticleGrid{1}
    # --- Persistent State (SoA) ---
    positions::Vector{Float64}
    rhos::Vector{Float64}
    curvatures::Vector{Float64}
    is_boundary::BitVector # BitVector is more memory efficient for booleans
    volumes::Vector{Float64}
    mood_events::BitVector

    # --- Pre-computed Coefficients (Ragged Arrays) ---
    neighbor_indices::Vector{Vector{Int}}
    # Note: The specific coefficient vectors (alfaij, etc.) are now part of the
    # interpolator's workspace, not the grid.

    # --- Grid Properties ---
    xmin::Float64
    xmax::Float64
    N::Int # Total number of particles
    dx::Float64
    regular::Bool
    bc::Symbol
    interior_indices::UnitRange{Int}
    max_volume::Ref{Float64}

    function ParticleGrid1D(
        xmin::Real, xmax::Real, N_interior::Integer, N_ghost::Integer, bc::Symbol; 
        randomness::Real = 0.0, rng = Meshfree4ScalarEq.rng
    )
        local dx
        if bc == :periodic
            @assert N_ghost == 0 "Periodic grids do not use ghost cells."
            N_total = N_interior
            interior_indices = 1:N_total
            dx = (xmax - xmin) / (N_interior > 1 ? (N_interior) : 1.0)
        else
            @assert N_ghost > 0 "N_ghost must be positive for non-periodic BCs."
            N_total = N_interior + 2 * N_ghost
            interior_indices = (N_ghost + 1):(N_ghost + N_interior)
            dx = (xmax - xmin) / (N_interior > 1 ? (N_interior - 1) : 1.0)
        end

        # --- Initialize SoA Fields ---
        positions = Vector{Float64}(undef, N_total)
        is_boundary = falses(N_total)
        
        # --- Populate Particle Positions ---
        
        
        if bc == :periodic
            # Create periodic interior points
            for i in 1:N_interior
                positions[i] = xmin + dx*(i-0.5) + randomness*(rand(rng, Float64)*2 - 1)
            end
        else
            # Left Ghosts
            for i in 1:N_ghost
                positions[i] = xmin - (N_ghost - i + 1) * dx
                is_boundary[i] = true
            end
            # Interior
            for i in 1:N_interior
                base_pos = (N_interior == 1) ? (xmin+xmax)/2.0 : xmin + (i-1) * dx
                positions[N_ghost + i] = base_pos + randomness*(rand(rng, Float64)*2 - 1)
            end
            # Right Ghosts
            for i in 1:N_ghost
                positions[N_ghost + N_interior + i] = xmax + i * dx
                is_boundary[N_ghost + N_interior + i] = true
            end
        end
        
        # --- Initialize Other Fields ---
        rhos = zeros(Float64, N_total)
        curvatures = zeros(Float64, N_total)
        volumes = zeros(Float64, N_total)
        mood_events = falses(N_total)
        neighbor_indices = [Int[] for _ in 1:N_total] # Initialize empty ragged array
        temp = zeros(Float64, N_total)
        regular = (randomness == 0.0)
        new(positions, rhos, curvatures, is_boundary, volumes, mood_events,
            neighbor_indices, xmin, xmax, N_total, dx, regular, bc, 
            interior_indices, Ref(0.))
    end
end


#==============================================================================
  2D PARTICLE GRID (Struct of Arrays Implementation)
==============================================================================#

struct ParticleGrid2D{S} <: ParticleGrid{2}
    positions::Vector{SVector{2, Float64}}
    rhos::Vector{Float64}
    curvatures::Matrix{Float64}
    is_boundary::BitVector
    volumes::Vector{Float64}
    voxels::Vector{Int}
    mood_events::BitVector
    neighbor_indices::Vector{Int}
    neighbor_pointers::Vector{Int}
    num_neighbors::Vector{Int}
    neighbor_xdistance::Vector{Float64}
    neighbor_ydistance::Vector{Float64}
    neighbor_df::Vector{Float64}      # NEW: For pre-calculated differences
    neighbor_weights::Vector{Float64} # NEW: For pre-calculated weights
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    Nx_total::Int
    Ny_total::Int
    N::Int
    N_ghost::Int
    dx::Float64
    dy::Float64
    regular::Bool
    bc::Symbol
    interior_indices::Vector{Int}
    neighbor_system::S;

    function ParticleGrid2D(
        xmin::Real, xmax::Real, ymin::Real, ymax::Real, 
        Nx_interior::Integer, Ny_interior::Integer, N_ghost::Integer, bc::Symbol, interp_range_factor::Real; 
        randomness::NTuple{2, Real} = (0.0, 0.0), rng = Meshfree4ScalarEq.rng
    )
        local Nx_total, Ny_total, interior_indices, dx_nominal, dy_nominal
        if bc == :periodic
            @assert N_ghost == 0 "Periodic grids do not use ghost cells."
            Nx_total, Ny_total = Nx_interior, Ny_interior
            interior_indices = collect(1:(Nx_interior * Ny_interior))
            dx_nominal = (xmax - xmin) / Nx_interior
            dy_nominal = (ymax - ymin) / Ny_interior
        else
            @assert N_ghost > 0 "N_ghost must be positive for non-periodic BCs."
            Nx_total = Nx_interior + 2*N_ghost
            Ny_total = Ny_interior + 2*N_ghost
            interior_indices = Int[]
            dx_nominal = (xmax - xmin) / (Nx_interior > 1 ? Nx_interior - 1 : 1.0)
            dy_nominal = (ymax - ymin) / (Ny_interior > 1 ? Ny_interior - 1 : 1.0)
        end
        N_total = Nx_total * Ny_total

        positions = Vector{SVector{2, Float64}}(undef, N_total)
        is_boundary = falses(N_total)
        interp_range = interp_range_factor * max(dx_nominal, dy_nominal)  
        # --- Populate Particle Positions ---
        if bc == :periodic
            # CORRECTED: Use cell-centered positions for periodic case
            for i in 1:Nx_total, j in 1:Ny_total
                index = (i - 1) * Ny_total + j
                posX = xmin + dx_nominal*(i-0.5) + randomness[1]*(rand(rng, Float64)*2 - 1)
                posY = ymin + dy_nominal*(j-0.5) + randomness[2]*(rand(rng, Float64)*2 - 1)
                positions[index] = SVector(posX, posY)
            end
            system = InPlaceNeighborList(
                x=positions, 
                cutoff=interp_range, 
                unitcell=[xmax-xmin; ymax-ymin],
                parallel=false # Enable parallelization
            )
        else # Non-periodic logic
            for i in 1:Nx_total, j in 1:Ny_total
                index = (i - 1) * Ny_total + j
                is_interior = (N_ghost < i <= Nx_interior + N_ghost) && (N_ghost < j <= Ny_interior + N_ghost)
                
                posX = if i <= N_ghost; xmin - (N_ghost-i+1)*dx_nominal; elseif i > Nx_interior+N_ghost; xmax+(i-(Nx_interior+N_ghost))*dx_nominal; else xmin+(i-N_ghost-1)*dx_nominal + randomness[1]*(rand(rng,Float64)*2-1); end
                posY = if j <= N_ghost; ymin - (N_ghost-j+1)*dy_nominal; elseif j > Ny_interior+N_ghost; ymax+(j-(Ny_interior+N_ghost))*dy_nominal; else ymin+(j-N_ghost-1)*dy_nominal + randomness[2]*(rand(rng,Float64)*2-1); end
                
                positions[index] = SVector(posX, posY)
                is_boundary[index] = !is_interior
                if is_interior; push!(interior_indices, index); end
            end
            system = InPlaceNeighborList(
                x=positions, 
                cutoff=interp_range, 
                parallel=false # Enable parallelization
            )
        end


        # 3. Create the stateful object

        new{typeof(system)}(positions, zeros(N_total), zeros(N_total, 2), is_boundary, zeros(N_total),
        zeros(Int, N_total), falses(N_total), Int[], zeros(Int,N_total+1), zeros(Int,N_total), Float64[], Float64[], Float64[], Float64[],
        xmin, xmax, ymin, ymax, Nx_total, Ny_total, N_total, N_ghost,
        dx_nominal, dy_nominal, (randomness == (0.0, 0.0)), bc, interior_indices, system)

    end
end


#==============================================================================
  Grid Functions (Optimized for SoA)
==============================================================================#



# --- Distance Functions ---
function getDistance(pg::ParticleGrid1D, i::Integer, j::Integer)
    dist = pg.positions[j] - pg.positions[i]
    if pg.bc == :periodic
        domainSize = pg.xmax - pg.xmin
        return dist - round(dist / domainSize) * domainSize
    else
        return dist
    end
end

function getDistance(pg::ParticleGrid2D{S}, i::Integer, j::Integer) where S
    return (pg.neighbor_xdistance[i][j], pg.neighbor_ydistance[i][j])
end
# function getDistance(pg::ParticleGrid2D, i::Integer, j::Integer)
#     dist_x = pg.positions[j][1] - pg.positions[i][1]
#     dist_y = pg.positions[j][2] - pg.positions[i][2]
#     if pg.bc == :periodic
#         domainSizeX = pg.xmax - pg.xmin
#         domainSizeY = pg.ymax - pg.ymin
#         dist_x -= round(dist_x / domainSizeX) * domainSizeX
#         dist_y -= round(dist_y / domainSizeY) * domainSizeY
#     end
#     return (dist_x, dist_y)
# end

function getDistance(pg::ParticleGrid2D{S}, x::SVector, y::SVector) where S
    dist_x = y[1] - x[1]
    dist_y = y[2] - x[2]
    if pg.bc == :periodic
        domainSizeX = pg.xmax - pg.xmin
        domainSizeY = pg.ymax - pg.ymin
        dist_x -= round(dist_x / domainSizeX) * domainSizeX
        dist_y -= round(dist_y / domainSizeY) * domainSizeY
    end
    return (dist_x, dist_y)
end

getEuclideanDistance(pg::ParticleGrid1D, i, j) = abs(getDistance(pg, i, j))
getEuclideanDistance(pg::ParticleGrid2D{S}, i, j) where S = norm(getDistance(pg, i, j)) 


# --- Voxel and Neighbor Search Helpers (Internal) ---
_grid_to_linear_index(hBox, vBox, nbBoxesX) = hBox + nbBoxesX * vBox
_linear_index_to_grid(linearIndex, nbBoxesX) = (mod(linearIndex, nbBoxesX), div(linearIndex, nbBoxesX))

"""
    updateNeighbors!(pg::ParticleGrid2D, weightFunc::MLSWeightFunction)

Builds and populates all flattened neighbor-list buffers in the particle grid.
This is the core function for consolidating slow, scattered memory reads into
a single, upfront, and efficient process.

It uses a two-pass algorithm to avoid allocations in the hot loop. In the
second pass, it computes and stores:
- Neighbor indices (`neighbor_indices`)
- x and y distances (`neighbor_xdistance`, `neighbor_ydistance`)
- The raw solution value of each neighbor (`neighbor_rhos`)
- The MLS weight for the interaction (`neighbor_weights`)
"""
function updateNeighbors!(pg::ParticleGrid2D{S}, weightFunc::MLSWeightFunction) where S
    system = pg.neighbor_system
    fVec = pg.rhos # Get a handle to the solution vector

    # --- PASS 1: COUNT NEIGHBORS ---
    # This pass is very fast as it only does additions.
    pg.num_neighbors .= 0
    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            pg.num_neighbors[i] += 1
            pg.num_neighbors[j] += 1
            null
        end,
        0, system.box, system.cl
    )

    # --- PREPARE FOR PASS 2 ---
    total_neighbors = sum(pg.num_neighbors)
    
    # Resize all flat arrays ONCE to the exact required size.
    resize!(pg.neighbor_indices, total_neighbors)
    resize!(pg.neighbor_xdistance, total_neighbors)
    resize!(pg.neighbor_ydistance, total_neighbors)
    resize!(pg.neighbor_weights, total_neighbors)
    resize!(pg.neighbor_df, total_neighbors)
    # Note: neighbor_df is NOT resized here, as it's calculated later.
    
    # Build the pointer array for fast indexing.
    pg.neighbor_pointers[1] = 1
    @inbounds for i in 1:pg.N
        pg.neighbor_pointers[i+1] = pg.neighbor_pointers[i] + pg.num_neighbors[i]
    end

    # Reset num_neighbors to be used as a per-particle offset counter in Pass 2.
    pg.num_neighbors .= 0
    
    # --- PASS 2: FILL ALL DATA ---
    # This pass performs all the necessary calculations and scattered reads.
    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            # --- 1. Perform scattered reads for solution values ---
            f_i = fVec[i]
            offset_i = pg.num_neighbors[i]
            write_idx_i = pg.neighbor_pointers[i] + offset_i
            pg.num_neighbors[i] += 1

            f_j = fVec[j]
            offset_j = pg.num_neighbors[j]
            write_idx_j = pg.neighbor_pointers[j] + offset_j
            pg.num_neighbors[j] += 1
            

            # --- 2. Calculate distances and weight ---
            dist_x = xj[1] - xi[1]
            dist_y = xj[2] - xi[2]
            if pg.bc == :periodic
                domainSizeX = pg.xmax - pg.xmin
                domainSizeY = pg.ymax - pg.ymin
                dist_x -= round(dist_x / domainSizeX) * domainSizeX
                dist_y -= round(dist_y / domainSizeY) * domainSizeY
            end
            weight = weightFunc(d2)

            # --- 3. Fill data for pair (i, j) ---
            # This uses the "base + offset" pattern which is ideal for the CPU prefetcher.
            diff = f_j - f_i
            
            pg.neighbor_indices[write_idx_i]   = j
            pg.neighbor_xdistance[write_idx_i] = dist_x
            pg.neighbor_ydistance[write_idx_i] = dist_y
            pg.neighbor_df[write_idx_i]        = diff
            pg.neighbor_weights[write_idx_i]   = weight
            

            # --- 4. Fill data for symmetric pair (j, i) ---


            pg.neighbor_indices[write_idx_j]   = i
            pg.neighbor_xdistance[write_idx_j] = -dist_x
            pg.neighbor_ydistance[write_idx_j] = -dist_y
            pg.neighbor_df[write_idx_j]        = -diff
            pg.neighbor_weights[write_idx_j]   = weight
            
            
            null
        end,
        0, system.box, system.cl
    )
    return nothing
end


"""
    set_df!(pg::ParticleGrid2D)

Pre-calculates the difference `f[neighbor] - f[particle]` for every neighbor
interaction and stores it in the `pg.neighbor_df` flat array. This moves
slow, scattered memory reads out of hot loops.
"""
function set_df!(pg::ParticleGrid2D{S}) where S
    fVec = pg.rhos
    # Loop over each particle in the grid
    for i in 1:pg.N
        num_nb = pg.num_neighbors[i]
        if num_nb == 0; continue; end

        # Get the starting index and value for the current particle
        start_idx = pg.neighbor_pointers[i]
        f_i = fVec[i]

        # Loop through this particle's neighbors
        @inbounds for k in 1:num_nb
            global_idx = start_idx + k - 1
            nb_idx = pg.neighbor_indices[global_idx]
            
            # Perform the scattered read here, ONCE.
            f_nb = fVec[nb_idx]
            
            # Store the result in the contiguous flat array.
            pg.neighbor_df[global_idx] = f_nb - f_i
        end
    end
    return nothing
end

function updateNeighbors!(particleGrid::ParticleGrid2D{S}) where S #, inner_radius::Real)
    system = particleGrid.neighbor_system
    box = cl.box
    outer_radius = box.cutoff # Get the radius from the box
    
    inner_radius = min(particleGrid.dx, particleGrid.dy)
    inner_radius_sq = inner_radius^2
    outer_radius_sq = outer_radius^2

    # Clear old neighbor lists
    for nb_list in particleGrid.neighbor_indices; empty!(nb_list); end

    # Update the cell list in-place with the current particle positions
    update_cell_list!(cl, particleGrid.positions, box)

    # Find neighbors (logic is the same)
    map_pairwise!(
        (i, j, d2, neighbor_lists) -> begin
            if inner_radius_sq <= d2 <= outer_radius_sq
                push!(neighbor_lists[i], j)
                push!(neighbor_lists[j], i)
            end
        end,
        particleGrid.neighbor_indices,
        system.box,
        system.cl
    )
    
    return nothing
end

"""
Optimized `updateNeighbors!` for 1D grids.
Assumes a sorted grid and handles periodic/non-periodic cases.
"""
function updateNeighbors!(particleGrid::ParticleGrid1D, maxDist::Real)
    N = particleGrid.N
    positions = particleGrid.positions
    domain_size = particleGrid.xmax - particleGrid.xmin

    for i in 1:N
        # Reuse the memory of the neighbor list for particle `i`
        nb_list = particleGrid.neighbor_indices[i]
        empty!(nb_list)
        pos_i = positions[i]

        if particleGrid.bc == :periodic
            # Search left, wrapping around the boundary
            for j_offset in 1:div(N, 2)
                j = mod1(i - j_offset, N)
                dist = abs(getDistance(particleGrid, i, j))
                if dist <= maxDist
                    push!(nb_list, j)
                else
                    break # Particles are sorted, so no need to check further
                end
            end
            # Search right, wrapping around the boundary
            for j_offset in 1:div(N, 2)
                j = mod1(i + j_offset, N)
                dist = abs(getDistance(particleGrid, i, j))
                if dist <= maxDist
                    push!(nb_list, j)
                else
                    break
                end
            end
        else # Non-periodic: search bounded by 1 and N
            # Search left
            for j in (i-1):-1:1
                if abs(positions[j] - pos_i) <= maxDist
                    push!(nb_list, j)
                else
                    break
                end
            end
            # Search right
            for j in (i+1):N
                if abs(positions[j] - pos_i) <= maxDist
                    push!(nb_list, j)
                else
                    break
                end
            end
        end
    end
end

"""
Calculates the 1D 'volume' (length of the Voronoi cell) for each particle.
"""
function determineVolumes!(particleGrid::ParticleGrid1D)
    N = particleGrid.N
    if N == 0; return; end

    positions = particleGrid.positions
    volumes = particleGrid.volumes
    
    if particleGrid.bc == :periodic
        for i in 1:N
            prev_idx = mod1(i - 1, N)
            next_idx = mod1(i + 1, N)
            # Use getDistance to correctly handle wrapping for edge particles
            deltaPosL = abs(getDistance(particleGrid, i, prev_idx))
            deltaPosR = abs(getDistance(particleGrid, i, next_idx))
            volumes[i] = (deltaPosL + deltaPosR) / 2.0
        end
    else
        # For non-periodic, only calculate for interior points
        for i in particleGrid.interior_indices
            volumes[i] = (positions[i+1] - positions[i-1]) / 2.0
        end
    end
    particleGrid.max_volume[] = maximum(volumes)
    return
end

"""
Updates the values in the ghost cells based on the grid's `bc` type for a 1D grid.
"""
function apply_boundary_conditions!(particleGrid::ParticleGrid1D)
    if particleGrid.bc != :outflow; return; end
    
    interior = particleGrid.interior_indices
    if isempty(interior); return; end
    
    first_interior_idx = first(interior)
    last_interior_idx  = last(interior)

    val_at_left_boundary  = particleGrid.rhos[first_interior_idx]
    val_at_right_boundary = particleGrid.rhos[last_interior_idx]

    # For 1D, slicing is correct and efficient.
    particleGrid.rhos[1:(first_interior_idx-1)] .= val_at_left_boundary
    particleGrid.rhos[(last_interior_idx+1):end] .= val_at_right_boundary
end


"""
Updates the values in the ghost cells based on the grid's `bc` type for a 2D grid.
"""
function apply_boundary_conditions!(particleGrid::ParticleGrid2D{S}) where S
    if particleGrid.bc != :outflow; return; end
    Nx = particleGrid.Nx_total
    Ny = particleGrid.Ny_total
    Ng = particleGrid.N_ghost
    
    # Iterate through all particles in the grid
    for i in 1:Nx, j in 1:Ny
        linear_idx = (i - 1) * Ny + j
        
        # Check if the particle is a ghost cell
        is_ghost = (i <= Ng || i > Nx - Ng || j <= Ng || j > Ny - Ng)

        if is_ghost
            # Find the closest interior column/row
            closest_i = clamp(i, Ng + 1, Nx - Ng)
            closest_j = clamp(j, Ng + 1, Ny - Ng)
            
            # Get the linear index of the nearest interior particle
            interior_idx = (closest_i - 1) * Ny + closest_j
            
            # Copy the value from that interior particle
            particleGrid.rhos[linear_idx] = particleGrid.rhos[interior_idx]
        end
    end
end

"""
Finds the local min/max in the neighborhood of a particle.
"""
function findLocalExtrema!(particleGrid::ParticleGrid, particleIndex::Integer, fVec::AbstractVector{Float64})
    # This function now works for both 1D and 2D without changes
    mini = fVec[particleIndex]
    maxi = fVec[particleIndex]
    for i in (particleGrid.neighbor_indices[particleIndex])
        mini = min(mini, fVec[i])
        maxi = max(maxi, fVec[i])
    end
    return (mini, maxi)
end

"""
    getTimeStep(particleGrid::ParticleGrid1D, eq::LinearAdvection{1}, interpAlpha::Real, interpRange::Real)

Return the maximum time step for which the first-order Euler & upwind method is a positive scheme for the 1D linear advection equation.
"""
function getTimeStep(particleGrid::ParticleGrid1D, eq::LinearAdvection{1}, interpAlpha::Real, interpRange::Real)
    dtMax = Inf
    # This function requires neighbor info, so we must update it first.
    updateNeighbors!(particleGrid, interpRange)
    
    # Ensure velocity from LinearAdvection{1} is a scalar
    vel = velocity(eq, 0.0)

    for particleIndex in particleGrid.interior_indices
        num = 0.0
        denum = 0.0
        for nbIndex in particleGrid.neighbor_indices[particleIndex]
            dx = getDistance(particleGrid, particleIndex, nbIndex)
            
            if ((vel >= 0.0) && (dx <= 0.0)) || ((vel <= 0.0) && (dx >= 0.0))
                w = exp(-interpAlpha * (dx^2))
                num += w * dx
                denum += w * dx * dx
            end
        end
        # Avoid division by zero if `num` is zero
        if abs(vel * num) > 1e-14
            dtMax = min(-denum / (vel * num), dtMax)
        end
    end
    return dtMax
end
function getTimeStep(particleGrid::ParticleGrid2D{S}, eq::LinearAdvection{2}) where S
    dtMax = Inf
    # Note: It's assumed that `updateNeighbors!` has already been called
    # and has populated all the neighbor_* buffers, including neighbor_weights.

    vel = eq.vel

    for particleIndex in particleGrid.interior_indices
        num_nb = particleGrid.num_neighbors[particleIndex]
        if num_nb == 0; continue; end
        
        start_idx = particleGrid.neighbor_pointers[particleIndex]

        # --- 1. First Pass: Calculate the least-squares matrix A ---
        # This loop reads directly from the global grid buffers. No allocations.
        A11 = 0.0; A12 = 0.0; A22 = 0.0
        @inbounds for k in 1:num_nb
            global_idx = start_idx + k - 1
            
            # Read pre-calculated values directly from the grid
            dx = particleGrid.neighbor_xdistance[global_idx]
            dy = particleGrid.neighbor_ydistance[global_idx]
            w  = particleGrid.neighbor_weights[global_idx]

            # Accumulate for the A matrix
            A11 += w * dx * dx
            A12 += w * dx * dy
            A22 += w * dy * dy
        end

        D = A11 * A22 - (A12^2)
        if abs(D) < 1e-14; continue; end

        # --- 2. Second Pass: Calculate sumCij using the A matrix ---
        # This loop also reads directly from the global grid buffers. No allocations.
        sumCij = 0.0
        @inbounds for k in 1:num_nb
            global_idx = start_idx + k - 1

            # Read pre-calculated values again
            deltaX = particleGrid.neighbor_xdistance[global_idx]
            deltaY = particleGrid.neighbor_ydistance[global_idx]
            w      = particleGrid.neighbor_weights[global_idx]
            
            # Solve 2x2 LS system for coefficients
            coeff_x = (A22 * w * deltaX - A12 * w * deltaY) / D
            coeff_y = (A11 * w * deltaY - A12 * w * deltaX) / D
            
            # Compute adapted coefficients for positivity
            angle = atan(deltaY, deltaX)
            n_x, n_y = cos(angle), sin(angle)
            s_x, s_y = -n_y, n_x
            
            alfaBar = n_x * coeff_x + n_y * coeff_y
            betaBar = s_x * coeff_x + s_y * coeff_y
            
            dot_vel_n = vel[1] * n_x + vel[2] * n_y
            dot_vel_s = vel[1] * s_x + vel[2] * s_y
            
            bracketMinus = dot_vel_n > 0.0 ? 0.0 : dot_vel_n
            bracketMinus2 = betaBar * dot_vel_s > 0.0 ? 0.0 : betaBar * dot_vel_s
            
            sumCij -= alfaBar * bracketMinus + bracketMinus2
        end
        
        if abs(sumCij) > 1e-14
            dtMax = min(1 / (2 * sumCij), dtMax)
        end
    end
    return dtMax
end

"""
Finds the local min/max and absolute min/max in the neighborhood of a particle
for a 1D vector of data `fVec`. Optimized for SoA grids.
"""
function findLocalExtremaAbs!(
    particleGrid::ParticleGrid1D, 
    particleIndex::Integer, 
    fVec::AbstractVector{Float64}
)::Tuple{Float64, Float64, Float64, Float64}
    
    val_i = fVec[particleIndex]
    mini = maxi = val_i
    minAbs = maxAbs = abs(val_i)
    
    # Access the neighbor list directly from the grid's SoA field
    for i in particleGrid.neighbor_indices[particleIndex]
        val_j = fVec[i]
        abs_val_j = abs(val_j)

        mini = min(mini, val_j)
        maxi = max(maxi, val_j)
        minAbs = min(minAbs, abs_val_j)
        maxAbs = max(maxAbs, abs_val_j)
    end
    
    return (mini, maxi, minAbs, maxAbs)
end

"""
Finds the local min/max and absolute min/max in the neighborhood of a particle
for a 2D matrix of data `fMatrix` (e.g., curvatures). Optimized for SoA grids.
"""
function findLocalExtremaAbs!(
    particleGrid::ParticleGrid2D{S}, 
    particleIndex::Integer, 
    fMatrix::AbstractMatrix{Float64}
)::NTuple{8, Float64} where S
    
    @assert size(fMatrix, 2) == 2 "Input matrix must have 2 columns for 2D extrema."

    # Initialize with values at the central particle
    val1_i = fMatrix[particleIndex, 1]
    val2_i = fMatrix[particleIndex, 2]
    
    mini1 = maxi1 = val1_i
    mini2 = maxi2 = val2_i
    
    minAbs1 = maxAbs1 = abs(val1_i)
    minAbs2 = maxAbs2 = abs(val2_i)
    
    # Access the neighbor list directly from the grid's SoA field
    for i in particleGrid.neighbor_indices[particleIndex]
        val1_j = fMatrix[i, 1]
        val2_j = fMatrix[i, 2]
        abs_val1_j = abs(val1_j)
        abs_val2_j = abs(val2_j)

        # Update extrema for the first component
        mini1 = min(mini1, val1_j)
        maxi1 = max(maxi1, val1_j)
        minAbs1 = min(minAbs1, abs_val1_j)
        maxAbs1 = max(maxAbs1, abs_val1_j)
        
        # Update extrema for the second component
        mini2 = min(mini2, val2_j)
        maxi2 = max(maxi2, val2_j)
        minAbs2 = min(minAbs2, abs_val2_j)
        maxAbs2 = max(maxAbs2, abs_val2_j)
    end
    
    return (mini1, maxi1, minAbs1, maxAbs1, mini2, maxi2, minAbs2, maxAbs2)
end



end  # module ParticleGrids