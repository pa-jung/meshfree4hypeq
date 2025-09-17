module ParticleGrids

export ParticleGrid, ParticleGrid1D, ParticleGrid2D, setInitialConditions!, getPeriodicDistance, saveGrid, plotDensity, 
       animateDensity, getTimeStep, findLocalExtrema!, updateVoxelInformation!, gridToLinearIndex, linearIndexToGrid, 
       findNeighbouringVoxels, updateNeighbours!, getEuclideanDistance, logMOODEvents!, findLocalExtremaAbs!, 
       determineVolumes!, getDistance, apply_boundary_conditions!, ParticleGridSystem

using FileIO, JLD2
using Plots
using Printf
using LaTeXStrings
using Statistics
using LinearAlgebra
using DelaunayTriangulation
using ..SimSettings
using ..HyperbolicPDEs
import Meshfree4ScalarEq

# --- Export new types and functions ---
export ParticleGrid, ParticleGrid1D, ParticleGrid2D, setInitialConditions!, 
       getDistance, saveGrid, plotDensity, animateDensity, getTimeStep, 
       findLocalExtrema!, updateNeighbours!, determineVolumes!, 
       apply_boundary_conditions!, ParticleGridSystem

# --- Core Abstract Type and System Alias ---
abstract type ParticleGrid{D} end # Now parameterized by dimension
const ParticleGridSystem{N, D} = NTuple{N, <:ParticleGrid{D}}


"""
    calculate_voronoi_volumes_2d(points::Vector{<:NTuple{2, Real}}, xmin, xmax, ymin, ymax) -> Vector{Float64}

Computes the area of the Voronoi cell for each point in `points` within a specified
bounding box, conforming to the API of DelaunayTriangulation.jl.

This implementation robustly handles cases where particles may lie exactly on the
boundary corners by reusing their indices instead of creating duplicate points.

# Arguments
- `points`: A vector of 2D particle locations, e.g., `[(x1, y1), (x2, y2), ...]`.
- `xmin`, `xmax`, `ymin`, `ymax`: The coordinates defining the bounding box.

# Returns
- A `Vector{Float64}` where the i-th element is the area of the Voronoi cell for the i-th input particle.
"""
function calculate_voronoi_volumes_2d(points::Vector{<:NTuple{2, Real}}, xmin, xmax, ymin, ymax)
    num_particles = length(points)
    
    # Create a mutable copy of the points to potentially add corners.
    all_points = [p for p in points] 

    # 1. Define the four corner points of the bounding box (CCW order).
    boundary_corners = [
        (xmin, ymin), # Lower-Left
        (xmax, ymin), # Lower-Right
        (xmax, ymax), # Upper-Right
        (xmin, ymax)  # Upper-Left
    ]

    # 2. Build the list of boundary INDICES.
    #    If a corner point already exists as a particle, reuse its index.
    #    Otherwise, add the corner to the master list of points and use its new index.
    boundary_indices = Int[]
    for corner_point in boundary_corners
        # `findfirst` is a robust way to check for existing points.
        idx = findfirst(p -> p == corner_point, all_points)
        if isnothing(idx)
            push!(all_points, corner_point)
            push!(boundary_indices, length(all_points))
        else
            push!(boundary_indices, idx)
        end
    end

    # 3. CRUCIAL: Close the loop by repeating the first index.
    push!(boundary_indices, boundary_indices[1])

    # 4. Triangulate using the combined points and the correctly formatted boundary indices.
    tri = triangulate(all_points; boundary_nodes = boundary_indices)

    # 5. Compute the Voronoi tessellation.
    vorn = voronoi(tri)

    # 6. Calculate the area for each Voronoi cell for the original particles.
    volumes = zeros(Float64, num_particles)
    for i in 1:num_particles
        volumes[i] = get_area(vorn, i)
    end
    
    return volumes
end



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
    neighbour_indices::Vector{Vector{Int}}
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
    
    # --- Temporary Buffers ---
    temp::Vector{Float64} # Can be used for various temporary calculations

    function ParticleGrid1D(
        xmin::Real, xmax::Real, N_interior::Integer, N_ghost::Integer, bc::Symbol; 
        randomness::Real = 0.0, rng = Meshfree4ScalarEq.rng
    )
        if bc == :periodic
            @assert N_ghost == 0 "Periodic grids do not use ghost cells."
            N_total = N_interior
            interior_indices = 1:N_total
        else
            @assert N_ghost > 0 "N_ghost must be positive for non-periodic BCs."
            N_total = N_interior + 2 * N_ghost
            interior_indices = (N_ghost + 1):(N_ghost + N_interior)
        end

        # --- Initialize SoA Fields ---
        positions = Vector{Float64}(undef, N_total)
        is_boundary = falses(N_total)
        
        # --- Populate Particle Positions ---
        dx = (xmax - xmin) / (N_interior > 1 ? (N_interior - 1) : 1.0)
        
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
        neighbour_indices = [Int[] for _ in 1:N_total] # Initialize empty ragged array
        temp = zeros(Float64, N_total)
        regular = (randomness == 0.0)

        new(positions, rhos, curvatures, is_boundary, volumes, mood_events,
            neighbour_indices, xmin, xmax, N_total, dx, regular, bc, 
            interior_indices, temp)
    end
end


#==============================================================================
  2D PARTICLE GRID (Struct of Arrays Implementation)
==============================================================================#

struct ParticleGrid2D <: ParticleGrid{2}
    # --- Persistent State (SoA) ---
    positions::Vector{NTuple{2, Float64}}
    rhos::Vector{Float64}
    curvatures::Matrix{Float64} # Stored as N x 2 matrix
    is_boundary::BitVector
    volumes::Vector{Float64}
    voxels::Vector{Int}
    mood_events::BitVector

    # --- Pre-computed Coefficients ---
    neighbour_indices::Vector{Vector{Int}}

    # --- Grid Properties ---
    xmin::Float64; xmax::Float64; ymin::Float64; ymax::Float64
    Nx_total::Int; Ny_total::Int; N_ghost::Int
    dx::Float64; dy::Float64
    regular::Bool; bc::Symbol
    interior_indices::Vector{Int}
    
    # --- Reusable Workspace for Neighbor Search ---
    voxel_map::Dict{Int, Vector{Int}}

    function ParticleGrid2D(
        xmin::Real, xmax::Real, ymin::Real, ymax::Real, 
        Nx_interior::Integer, Ny_interior::Integer, N_ghost::Integer, bc::Symbol; 
        randomness::NTuple{2, Real} = (0.0, 0.0), rng = Meshfree4ScalarEq.rng
    )
        if bc == :periodic
            @assert N_ghost == 0 "Periodic grids do not use ghost cells."
            Nx_total, Ny_total = Nx_interior, Ny_interior
        else
            @assert N_ghost > 0 "N_ghost must be positive for non-periodic BCs."
            Nx_total = Nx_interior + 2 * N_ghost
            Ny_total = Ny_interior + 2 * N_ghost
        end
        N_total = Nx_total * Ny_total

        # --- Initialize SoA Fields ---
        positions = Vector{NTuple{2, Float64}}(undef, N_total)
        is_boundary = falses(N_total)
        interior_indices = Int[]
        
        # --- Populate Particle Positions ---
        dx = (xmax - xmin) / (Nx_interior > 1 ? Nx_interior - 1 : 1.0)
        dy = (ymax - ymin) / (Ny_interior > 1 ? Ny_interior - 1 : 1.0)
        
        for i in 1:Nx_total, j in 1:Ny_total
            index = (i - 1) * Ny_total + j
            is_interior = (N_ghost < i <= Nx_interior + N_ghost) && (N_ghost < j <= Ny_interior + N_ghost)

            # Determine particle position
            local posX, posY
            # X-position
            if i <= N_ghost # Left ghosts
                posX = xmin - (N_ghost - i + 1) * dx
            elseif i > Nx_interior + N_ghost # Right ghosts
                posX = xmax + (i - (Nx_interior + N_ghost)) * dx
            else # Interior x-range
                base_posX = xmin + (i - N_ghost - 1) * dx
                posX = base_posX + randomness[1] * (rand(rng, Float64) * 2 - 1)
            end
            # Y-position
            if j <= N_ghost # Bottom ghosts
                posY = ymin - (N_ghost - j + 1) * dy
            elseif j > Ny_interior + N_ghost # Top ghosts
                posY = ymax + (j - (Ny_interior + N_ghost)) * dy
            else # Interior y-range
                base_posY = ymin + (j - N_ghost - 1) * dy
                posY = base_posY + randomness[2] * (rand(rng, Float64) * 2 - 1)
            end
            positions[index] = (posX, posY)
            is_boundary[index] = !is_interior
            if is_interior; push!(interior_indices, index); end
        end

        # --- Initialize Other Fields ---
        rhos = zeros(Float64, N_total)
        curvatures = zeros(Float64, N_total, 2)
        volumes = zeros(Float64, N_total)
        voxels = zeros(Int, N_total)
        mood_events = falses(N_total)
        neighbour_indices = [Int[] for _ in 1:N_total]
        voxel_map = Dict{Int, Vector{Int}}()
        regular = (randomness == (0.0, 0.0))

        new(positions, rhos, curvatures, is_boundary, volumes, voxels, mood_events,
            neighbour_indices, xmin, xmax, ymin, ymax, Nx_total, Ny_total, N_ghost,
            dx, dy, regular, bc, interior_indices, voxel_map)
    end
end


#==============================================================================
  Grid Functions (Optimized for SoA)
==============================================================================#

# --- Set Initial Conditions ---
function setInitialConditions!(particleGrid::ParticleGrid{D}, initFunc::Function) where D
    for i in 1:particleGrid.N
        # Pass position (Float64 for 1D, NTuple for 2D) to initFunc
        particleGrid.rhos[i] = initFunc(particleGrid.positions[i]...)
    end
end

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

function getDistance(pg::ParticleGrid2D, i::Integer, j::Integer)
    dist_x = pg.positions[j][1] - pg.positions[i][1]
    dist_y = pg.positions[j][2] - pg.positions[i][2]
    if pg.bc == :periodic
        domainSizeX = pg.xmax - pg.xmin
        domainSizeY = pg.ymax - pg.ymin
        dist_x -= round(dist_x / domainSizeX) * domainSizeX
        dist_y -= round(dist_y / domainSizeY) * domainSizeY
    end
    return (dist_x, dist_y)
end

getEuclideanDistance(pg::ParticleGrid1D, i, j) = abs(getDistance(pg, i, j))
getEuclideanDistance(pg::ParticleGrid2D, i, j) = norm(getDistance(pg, i, j))


# --- OPTIMIZED updateNeighbours! for 2D grids ---
function updateNeighbours!(particleGrid::ParticleGrid2D, maxDist::Real)
    # Voxel setup
    all_pos = particleGrid.positions
    domain_xmin, domain_xmax = isempty(all_pos) ? (0.0,0.0) : extrema(p[1] for p in all_pos)
    domain_ymin, domain_ymax = isempty(all_pos) ? (0.0,0.0) : extrema(p[2] for p in all_pos)
    
    nbBoxesX = floor(Int, (domain_xmax - domain_xmin) / maxDist) + 1
    nbBoxesY = floor(Int, (domain_ymax - domain_ymin) / maxDist) + 1

    # --- 1. Update Voxel Information ---
    xBoxSize = (domain_xmax - domain_xmin) / nbBoxesX
    yBoxSize = (domain_ymax - domain_ymin) / nbBoxesY
    for i in 1:particleGrid.N
        hBox = min(floor(Int, (particleGrid.positions[i][1] - domain_xmin) / xBoxSize), nbBoxesX - 1)
        vBox = min(floor(Int, (particleGrid.positions[i][2] - domain_ymin) / yBoxSize), nbBoxesY - 1)
        particleGrid.voxels[i] = (vBox * nbBoxesX) + hBox
    end

    # --- 2. Reuse and Refill Voxel Map ---
    voxel_map = particleGrid.voxel_map
    for key in keys(voxel_map); empty!(voxel_map[key]); end
    
    for i in 1:particleGrid.N
        voxel_idx = particleGrid.voxels[i]
        if !haskey(voxel_map, voxel_idx); voxel_map[voxel_idx] = Int[]; end
        push!(voxel_map[voxel_idx], i)
    end

    # --- 3. Find Neighbors ---
    for p_idx in 1:particleGrid.N
        # Reuse memory for the neighbor list
        empty!(particleGrid.neighbour_indices[p_idx])
        
        # ... (findNeighbouringVoxels logic is complex and assumed correct) ...
        # Simplified placeholder for the neighbor search logic:
        # You would call your `findNeighbouringVoxels` helper here.
        # For simplicity, this example just checks all particles.
        # REPLACE THIS with your more efficient voxel-based search.
        for nb_idx in 1:particleGrid.N
            if p_idx != nb_idx && getEuclideanDistance(particleGrid, p_idx, nb_idx) <= maxDist
                push!(particleGrid.neighbour_indices[p_idx], nb_idx)
            end
        end
    end
end

"""
    gridToLinearIndex(hBox::Integer, vBox::Integer, nbBoxesX::Integer)::Integer

Convert 2D grid index to a linear index.
"""
function gridToLinearIndex(hBox::Integer, vBox::Integer, nbBoxesX::Integer)::Integer
    return hBox + nbBoxesX*vBox
end


"""
    linearIndexToGrid(linearIndex::Integer, nbBoxesX::Integer)::Tuple{Integer, Integer}
    
Convert linear index to row and column index (hBox, vBox).
"""
function linearIndexToGrid(linearIndex::Integer, nbBoxesX::Integer)::Tuple{Integer, Integer}
    hBox = mod(linearIndex, nbBoxesX)
    vBox = div(linearIndex - hBox, nbBoxesX)
    return (hBox, vBox)
end

"""
Optimized `updateNeighbours!` for 1D grids.
Assumes a sorted grid and handles periodic/non-periodic cases.
"""
function updateNeighbours!(particleGrid::ParticleGrid1D, maxDist::Real)
    N = particleGrid.N
    positions = particleGrid.positions
    domain_size = particleGrid.xmax - particleGrid.xmin

    for i in 1:N
        # Reuse the memory of the neighbor list for particle `i`
        nb_list = particleGrid.neighbour_indices[i]
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
end

"""
Calculates the 2D 'volume' (area of the Voronoi cell) for each particle.
"""
function determineVolumes!(particleGrid::ParticleGrid2D)
    if particleGrid.N == 0; return; end
    
    # Bounding box must encompass all points, including ghosts
    xmin_b = particleGrid.xmin - (particleGrid.N_ghost + 0.5) * particleGrid.dx
    xmax_b = particleGrid.xmax + (particleGrid.N_ghost + 0.5) * particleGrid.dx
    ymin_b = particleGrid.ymin - (particleGrid.N_ghost + 0.5) * particleGrid.dy
    ymax_b = particleGrid.ymax + (particleGrid.N_ghost + 0.5) * particleGrid.dy

    # The `calculate_voronoi_volumes_2d` helper function is assumed to be defined
    # at the top of the module as in your original file.
    vols = calculate_voronoi_volumes_2d(particleGrid.positions, xmin_b, xmax_b, ymin_b, ymax_b)
    
    if length(vols) == particleGrid.N
        particleGrid.volumes .= vols
    else
        @warn "Voronoi cell calculation returned an incorrect number of volumes. Volumes not updated."
    end
end

"""
Updates the values in the ghost cells based on the grid's `bc` type.
"""
function apply_boundary_conditions!(particleGrid::ParticleGrid)
    if particleGrid.bc == :periodic || particleGrid.bc == :fixed_dirichlet
        return # No action needed
    elseif particleGrid.bc == :outflow
        interior = particleGrid.interior_indices
        if isempty(interior); return; end
        
        first_interior_idx = first(interior)
        last_interior_idx = last(interior)

        val_at_left_boundary = particleGrid.rhos[first_interior_idx]
        val_at_right_boundary = particleGrid.rhos[last_interior_idx]

        # Set left ghost particles (indices 1 to first_interior_idx - 1)
        particleGrid.rhos[1:(first_interior_idx-1)] .= val_at_left_boundary
        
        # Set right ghost particles (indices last_interior_idx + 1 to end)
        particleGrid.rhos[(last_interior_idx+1):end] .= val_at_right_boundary
    else
        error("Unsupported boundary condition type: $(particleGrid.bc)")
    end
end

"""
Finds the local min/max in the neighbourhood of a particle.
"""
function findLocalExtrema!(particleGrid::ParticleGrid, particleIndex::Integer, fVec::AbstractVector{Float64})
    # This function now works for both 1D and 2D without changes
    mini = fVec[particleIndex]
    maxi = fVec[particleIndex]
    for i in particleGrid.neighbour_indices[particleIndex]
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
    updateNeighbours!(particleGrid, interpRange)
    
    # Ensure velocity from LinearAdvection{1} is a scalar
    vel = velocity(eq, 0.0)

    for particleIndex in particleGrid.interior_indices
        num = 0.0
        denum = 0.0
        for nbIndex in particleGrid.neighbour_indices[particleIndex]
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

"""
    getTimeStep(particleGrid::ParticleGrid2D, eq::LinearAdvection{2}, interpAlpha::Real, interpRange::Real)

Return the maximum time step for which Praveen's upwind method is a positive scheme for the 2D linear advection equation.
"""
function getTimeStep(particleGrid::ParticleGrid2D, eq::LinearAdvection{2}, interpAlpha::Real, interpRange::Real)
    dtMax = Inf
    updateNeighbours!(particleGrid, interpRange)

    vel = velocity(eq, 0.0)

    for particleIndex in particleGrid.interior_indices
        # Create 2x2 LS system
        A11 = A12 = A22 = 0.0
        for nbIndex in particleGrid.neighbour_indices[particleIndex]
            deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
            w = exp(-interpAlpha * (deltaX^2 + deltaY^2) / (interpRange^2))
            A11 += w * (deltaX^2)
            A12 += w * deltaX * deltaY
            A22 += w * (deltaY^2)
        end
        D = A11 * A22 - (A12^2)
        
        if abs(D) < 1e-14; continue; end # Avoid division by zero if matrix is singular

        sumCij = 0.0
        for nbIndex in particleGrid.neighbour_indices[particleIndex]
            deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
            w = exp(-interpAlpha * (deltaX^2 + deltaY^2) / (interpRange^2))
            
            # Solve 2x2 LS system for coefficients
            coeff_x = (A22 * w * deltaX - A12 * w * deltaY) / D
            coeff_y = (A11 * w * deltaY - A12 * w * deltaX) / D
            
            # Compute adapted coefficients for positivity
            angle = atan(deltaY, deltaX)
            n = (cos(angle), sin(angle))
            s = (-sin(angle), cos(angle))
            
            alfaBar = n[1] * coeff_x + n[2] * coeff_y
            betaBar = s[1] * coeff_x + s[2] * coeff_y
            
            dot_vel_n = vel[1] * n[1] + vel[2] * n[2]
            dot_vel_s = vel[1] * s[1] + vel[2] * s[2]
            
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

end  # module ParticleGrids