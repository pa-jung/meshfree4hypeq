"""
    reset_voxels!(lv::LocalVoxels)

Clears the occupation buffer.
"""
function reset_voxels!(lv::LocalVoxels)
    fill!(lv.occupation, false)
    # Always mark the center bin (0-offset) as occupied by the particle itself
    # Index in 1-based array: half_bins + 1
    center_idx = lv.half_bins + 1
    lv.occupation[center_idx] = true
end

"""
    check_occupation!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int)

Iterates over neighbors of `i` and marks their corresponding relative voxels as occupied.
Also marks neighbors as 'visited' in the global buffer to prevent redundant checks.
"""
function check_occupation!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int, visited::BitVector)
    R = pg.max_dist
    start_ptr = pg.neighbor_pointers[i]
    num_nbs   = pg.num_neighbors[i]
    
    center_offset = lv.half_bins + 1

    for k in 0:(num_nbs - 1)
        flat_idx = start_ptr + k
        dist = pg.neighbor_xdistance[flat_idx]
        
        # 1. Check bounds (strictly within [-R, R])
        if abs(dist) > R; continue; end
        
        # 2. Calculate Bin Index
        # Relative index: round(dist / size)
        # We use floor to be robust: bin = floor(dist / size + 0.5)
        rel_idx = floor(Int, dist / lv.voxel_size + 0.5)
        
        bin_idx = center_offset + rel_idx
        
        # 3. Mark Occupied
        if bin_idx >= 1 && bin_idx <= lv.num_bins
            lv.occupation[bin_idx] = true
            
            # 4. Mark Global Visited (Optimization)
            # If a neighbor is in a valid voxel, we consider it "checked" for this pass
            nb_idx = pg.neighbor_indices[flat_idx]
            visited[nb_idx] = true 
        end
    end
end

# """
#     fill_empty_voxels!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int)

# Scans the voxel map. If a voxel is empty, generates a split particle
# at that voxel's center.
# """
# function fill_empty_voxels!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int)
#     center_offset = lv.half_bins + 1
    
#     for bin_idx in 1:lv.num_bins
#         if !lv.occupation[bin_idx]
#             # --- Found Empty Voxel ---
            
#             # 1. Calculate Relative Position
#             # rel_idx = bin_idx - center_offset
#             # pos = rel_idx * voxel_size
#             rel_idx = bin_idx - center_offset
#             rel_pos = rel_idx * lv.voxel_size
            
#             abs_pos = pg.positions[i] + rel_pos
            
#             # 2. Boundary / Domain Checks
#             if pg.bc == :periodic
#                 L = pg.xmax - pg.xmin
#                 if abs_pos > pg.xmax; abs_pos -= L; end
#                 if abs_pos < pg.xmin; abs_pos += L; end
#             else
#                 if abs_pos < pg.xmin || abs_pos > pg.xmax
#                     continue # Skip out-of-bounds voxels
#                 end
#             end
            
#             # 3. Interpolate Density
#             # Using the helper to look up neighbors relative to this new position
#             new_rho = _interpolate_rho_at_rel_pos(pg, i, rel_pos)
            
#             # 4. Push
#             push!(pg.split_buffer_pos, abs_pos)
#             push!(pg.split_buffer_rho, new_rho)
#         end
#     end
# end
"""
    fill_empty_voxels!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int)

Scans the `LocalVoxels` map. If a voxel is empty:
1. Identifies closest neighbors L and R.
2. If Bounded (L & R exist): Linear Interpolation.
3. If Outer (Missing L or R): Extrapolates from the closest neighbor 
   using its AVERAGED local slope.
"""
function fill_empty_voxels!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int)
    center_offset = lv.half_bins + 1
    
    for bin_idx in 1:lv.num_bins
        if !lv.occupation[bin_idx]
            # --- Found Empty Voxel ---
            
            # 1. Calculate Target Position
            rel_idx = bin_idx - center_offset
            rel_pos = rel_idx * lv.voxel_size
            abs_pos = pg.positions[i] + rel_pos
            
            # Domain Check
            if pg.bc == :periodic
                L = pg.xmax - pg.xmin
                if abs_pos > pg.xmax; abs_pos -= L; end
                if abs_pos < pg.xmin; abs_pos += L; end
            elseif abs_pos < pg.xmin || abs_pos > pg.xmax
                continue # Skip out-of-bounds
            end
            
            # 2. Find Closest Neighbors (Relative to New Position)
            closest_L_dist = -Inf; closest_L_idx = -1
            closest_R_dist = Inf;  closest_R_idx = -1
            
            # Check neighbors of center particle 'i'
            start_ptr = pg.neighbor_pointers[i]
            for n in 0:(pg.num_neighbors[i]-1)
                flat_idx = start_ptr + n
                d_from_i = pg.neighbor_xdistance[flat_idx]
                nb_idx   = pg.neighbor_indices[flat_idx]
                
                d_new = d_from_i - rel_pos # Dist from new particle
                
                if d_new < 0 && d_new > closest_L_dist
                    closest_L_dist = d_new; closest_L_idx = nb_idx
                elseif d_new > 0 && d_new < closest_R_dist
                    closest_R_dist = d_new; closest_R_idx = nb_idx
                end
            end
            
            # Check 'i' itself
            d_new_i = 0.0 - rel_pos
            if d_new_i < 0 && d_new_i > closest_L_dist
                closest_L_dist = d_new_i; closest_L_idx = i
            elseif d_new_i > 0 && d_new_i < closest_R_dist
                closest_R_dist = d_new_i; closest_R_idx = i
            end

            # 3. Calculate Rho using New Logic
            new_rho = 0.0
            
            if closest_L_idx != -1 && closest_R_idx != -1
                # Case A: Interior Voxel -> Linear Interpolation
                rho_L = pg.rhos[closest_L_idx]
                rho_R = pg.rhos[closest_R_idx]
                t = (0.0 - closest_L_dist) / (closest_R_dist - closest_L_dist)
                new_rho = rho_L * (1.0 - t) + rho_R * t
                
            elseif closest_L_idx == -1 && closest_R_idx != -1
                # Case B: Outer Voxel (Left Void) -> Extrapolate from Right
                # j is the closest neighbor to the Right
                j = closest_R_idx
                
                # Get averaged slope at j
                avg_slope = _get_average_slope(pg, j)
                
                # Extrapolate: rho_new = rho_j + slope * (pos_new - pos_j)
                # (pos_new - pos_j) is exactly -closest_R_dist
                new_rho = pg.rhos[j] + avg_slope * (-closest_R_dist)
                
            elseif closest_L_idx != -1 && closest_R_idx == -1
                # Case C: Outer Voxel (Right Void) -> Extrapolate from Left
                # j is the closest neighbor to the Left
                j = closest_L_idx
                
                # Get averaged slope at j
                avg_slope = _get_average_slope(pg, j)
                
                # Extrapolate: rho_new = rho_j + slope * (pos_new - pos_j)
                # (pos_new - pos_j) is exactly -closest_L_dist
                new_rho = pg.rhos[j] + avg_slope * (-closest_L_dist)
                
            else
                # Case D: Isolated (Fallback)
                new_rho = pg.rhos[i]
            end
            
            # 4. Commit
            push!(pg.split_buffer_pos, abs_pos)
            push!(pg.split_buffer_rho, new_rho)
        end
    end
end

"""
    _get_average_slope(pg, i)

Calculates the weighted average slope from particle `i` to all its neighbors.
Used for robust extrapolation into voids.
"""
function _get_average_slope(pg, i::Int)
    start_ptr = pg.neighbor_pointers[i]
    num_nbs   = pg.num_neighbors[i]
    
    if num_nbs == 0
        return 0.0 # Isolated particle, assume flat slope
    end
    
    sum_weighted_slope = 0.0
    sum_weights = 0.0
    rho_i = pg.rhos[i]
    
    for k in 0:(num_nbs-1)
        flat_idx = start_ptr + k
        
        # Retrieve neighbor data
        dx = pg.neighbor_xdistance[flat_idx] # pos_neighbor - pos_i
        nb_idx = pg.neighbor_indices[flat_idx]
        w  = pg.neighbor_weights[flat_idx]
        
        # Retrieve rho
        rho_nb = pg.rhos[nb_idx]
        
        # Compute Slope (d_rho / d_x)
        # Check tolerance to avoid division by zero
        if abs(dx) > 1e-12
            slope = (rho_nb - rho_i) / dx
            
            sum_weighted_slope += w * slope
            sum_weights += w
        end
    end
    
    if sum_weights < 1e-14
        return 0.0
    end
    
    return sum_weighted_slope / sum_weights
end
"""
    _interpolate_rho_at_rel_pos(pg, center_i, rel_pos)

Helper to find neighbors closest to `rel_pos` (relative to `center_i`)
and interpolate density. Falls back to Constant extrapolation at boundaries.
"""
function _interpolate_rho_at_rel_pos(pg::ParticleGrid1D, i::Int, rel_pos::Float64)
    # Find P_Left and P_Right relative to the *new* position
    # dist_from_new = dist_from_i - rel_pos
    
    closest_L_dist = -Inf
    closest_L_idx  = -1
    closest_R_dist = Inf
    closest_R_idx  = -1
    
    # 1. Check neighbors of i
    start_ptr = pg.neighbor_pointers[i]
    for n in 0:(pg.num_neighbors[i]-1)
        flat_idx = start_ptr + n
        d_from_i = pg.neighbor_xdistance[flat_idx]
        nb_idx   = pg.neighbor_indices[flat_idx]
        
        d_new = d_from_i - rel_pos
        
        if d_new < 0 && d_new > closest_L_dist
            closest_L_dist = d_new
            closest_L_idx  = nb_idx
        elseif d_new > 0 && d_new < closest_R_dist
            closest_R_dist = d_new
            closest_R_idx  = nb_idx
        end
    end
    
    # 2. Check i itself
    d_new_i = 0.0 - rel_pos
    if d_new_i < 0 && d_new_i > closest_L_dist
        closest_L_dist = d_new_i
        closest_L_idx  = i
    elseif d_new_i > 0 && d_new_i < closest_R_dist
        closest_R_dist = d_new_i
        closest_R_idx  = i
    end

    # 3. Interpolate
    if closest_L_idx != -1 && closest_R_idx != -1
        # Linear
        rho_L = pg.rhos[closest_L_idx]
        rho_R = pg.rhos[closest_R_idx]
        total = closest_R_dist - closest_L_dist
        t = (0.0 - closest_L_dist) / total
        return rho_L * (1.0 - t) + rho_R * t
    elseif closest_L_idx != -1
        return pg.rhos[closest_L_idx] # Constant Extrapolation from Left
    elseif closest_R_idx != -1
        return pg.rhos[closest_R_idx] # Constant Extrapolation from Right
    else
        return pg.rhos[i] # Fallback
    end
end

"""
    manage_particles!(pg::ParticleGrid1D)

Main management routine using a Voxel-Based Gap Filling strategy followed by Merging.
"""
function manage_particles!(pg::ParticleGrid1D)
# =========================================================================
    # PHASE 1: VOXEL FILL (Splitting)
    # =========================================================================
    empty!(pg.split_buffer_pos)
    empty!(pg.split_buffer_rho)

    # 1. Reset Global Visited Buffer
    if length(pg.merged_buffer) < pg.N
        safe_resize!(pg.merged_buffer, pg.N)
    end
    visited = pg.merged_buffer 
    fill!(view(visited, 1:pg.N), false)
    
    # 2. Access the LocalVoxels struct
    lv = pg.local_voxels

    # 3. Iterate
    for i in 1:pg.N
        if !visited[i]
            # Reset the local map for this particle
            reset_voxels!(lv)
            
            # Check neighbors to fill the map
            # This ALSO marks those neighbors as 'visited' in the global buffer
            check_occupation!(lv, pg, i, visited)
            
            # If map has holes, create particles
            fill_empty_voxels!(lv, pg, i)
            
            # Mark self as visited
            visited[i] = true
        end
    end

    # 3. Append New Particles
    N_new = length(pg.split_buffer_pos)
    println("Buffer-length: ", N_new)
    if N_new > 0
        N_curr = pg.N
        N_total = N_curr + N_new
        
        safe_resize!(pg.positions, N_total)
        safe_resize!(pg.rhos, N_total)
        safe_resize!(pg.curvatures, N_total)
        safe_resize!(pg.is_boundary, N_total)
        safe_resize!(pg.volumes, N_total)
        safe_resize!(pg.mood_events, N_total)
        safe_resize!(pg.neighbor_pointers, N_total + 1)
        safe_resize!(pg.num_neighbors, N_total)
        
        for k in 1:N_new
            idx = N_curr + k
            pg.positions[idx]   = pg.split_buffer_pos[k]
            pg.rhos[idx]        = pg.split_buffer_rho[k]
            
            # Defaults
            pg.curvatures[idx]  = 0.0
            pg.is_boundary[idx] = false
            pg.volumes[idx]     = 0.0
            pg.mood_events[idx] = false
        end
        
        pg.N = N_total
        
        # Intermediate Rebuild
        sort_1d_particles!(pg)
        updateNeighbors!(pg)
        determineVolumes!(pg)
    end
    println(pg.N)
    println(pg.split_buffer_pos)
    if pg.N > 2000; error("Too many particles!") end
    # =========================================================================
    # PHASE 2: MERGE (Coarsen)
    # =========================================================================
    
    # Reset buffer for merging (reuse same buffer)
    safe_resize!(pg.merged_buffer, pg.N)
    fill!(view(pg.merged_buffer, 1:pg.N), false)
    merged = pg.merged_buffer # Alias

    write_idx = 0 
    
    # Accessors
    pos = pg.positions
    rhos = pg.rhos
    is_bd = pg.is_boundary
    
    for i in 1:pg.N
        if merged[i]; continue; end

        write_idx += 1
        
        # Accumulators
        sum_x = pos[i]
        sum_rho = rhos[i]
        count = 1.0
        is_boundary_particle = is_bd[i]
        
        # Check Neighbors
        start_ptr = pg.neighbor_pointers[i]
        n_count   = pg.num_neighbors[i]
        
        if n_count > 0
            end_ptr = start_ptr + n_count - 1
            for k in start_ptr:end_ptr
                j = pg.neighbor_indices[k]
                
                # Merge Conditions: Future particle, not merged, close enough
                if j > i && !merged[j]
                    dist = abs(pg.neighbor_xdistance[k]) 
                    
                    if dist < pg.min_dist
                        sum_x += pos[j]
                        sum_rho += rhos[j]
                        count += 1.0
                        if is_bd[j]; is_boundary_particle = true; end
                        merged[j] = true
                    end
                end
            end
        end
        
        # Write compacted result
        pos[write_idx]   = sum_x / count
        rhos[write_idx]  = sum_rho / count
        is_bd[write_idx] = is_boundary_particle
    end

    # =========================================================================
    # PHASE 3: FINALIZE
    # =========================================================================

    pg.N = write_idx
    
    if pg.bc == :periodic
         pg.interior_indices = 1:pg.N
    else
         pg.interior_indices = (pg.N_ghost + 1):(pg.N - pg.N_ghost)
    end
    println(pg.N)
    #error("TEST")
    safe_resize!(pg.num_neighbors, pg.N)
    safe_resize!(pg.neighbor_pointers, pg.N + 1)
    sort_1d_particles!(pg)
    updateNeighbors!(pg)
    determineVolumes!(pg)
end

"""
    _voxel_fill_particle!(pg, center_idx, visited)

Creates a virtual voxel grid around `center_idx`.
Checks each voxel for existing particles.
- If found: Marks them as visited.
- If empty: Inserts a new particle (Linear Interp or Gradient Extrapolation).
"""
function _voxel_fill_particle!(pg::ParticleGrid1D, center_idx::Int, visited::BitVector)
    # 1. Define Local Grid
    # Range is usually the smoothing length (interp_range or max_dist)
    R = pg.max_dist
    
    # Voxel Size: We want 'min_nb' voxels per radius R to ensure density
    # Or simply: Total window is 2*R.
    # The user specified: voxel_len = interp_range / min_nb
    # Assuming max_dist approx interp_range
    # We use min_nb as the number of voxels per side (Left/Right).
    
    min_nb = max(pg.min_nb, 2) # Safety: ensure we have at least 2 bins
    voxel_size = R / (min_nb) # distribute coverage slightly wider
    
    # Mark Center as visited
    visited[center_idx] = true
    
    # 2. Iterate Voxels
    # We scan from -min_nb to +min_nb (covering roughly [-R, +R])
    # k=0 is the center voxel (where center_idx lives)
    
    start_ptr = pg.neighbor_pointers[center_idx]
    num_nbs   = pg.num_neighbors[center_idx]

    # Iterate integer chunks
    # Range: We want to cover [-R, R].
    # k goes from -min_nb to min_nb - 1
    for k in -Int(min_nb):(Int(min_nb)-1)
        
        # Voxel Bounds (Relative to center)
        v_start = k * voxel_size
        v_end   = (k + 1) * voxel_size
        
        # A. Check Occupancy
        occupied = false
        
        # Check Center Particle (k=0 usually covers 0.0)
        if v_start <= 0.0 && v_end > 0.0
            occupied = true
        end
        
        # Check Neighbors
        for n in 0:(num_nbs-1)
            flat_idx = start_ptr + n
            dist = pg.neighbor_xdistance[flat_idx] # Relative distance
            
            if dist >= v_start && dist < v_end
                occupied = true
                
                # Mark neighbor as visited so we don't process it again
                nb_idx = pg.neighbor_indices[flat_idx]
                visited[nb_idx] = true
            end
        end
        
        # B. If Empty -> Fill
        if !occupied
            _fill_empty_voxel(pg, center_idx, v_start, v_end)
        end
    end
end

"""
    _fill_empty_voxel(pg, i, v_start, v_end)

Calculates position and interpolated density for an empty voxel.
Uses Linear Interpolation if bounded, or Gradient Extrapolation if boundary.
"""
function _fill_empty_voxel(pg, i, v_start, v_end)
    # 1. New Position (Center of Voxel)
    rel_pos = (v_start + v_end) / 2.0
    abs_pos = pg.positions[i] + rel_pos
    
    # Domain Check (Strict)
    if pg.bc != :periodic
        if abs_pos < pg.xmin || abs_pos > pg.xmax
            return
        end
    end
    
    # Apply Periodic Wrap for calculation if needed
    if pg.bc == :periodic
        domain = pg.xmax - pg.xmin
        if abs_pos > pg.xmax; abs_pos -= domain; end
        if abs_pos < pg.xmin; abs_pos += domain; end
    end

    # 2. Find Nearest Neighbors (P_Left and P_Right) relative to voxel center
    # "Left" and "Right" here refer to direction relative to the new particle
    
    closest_L_dist = -Inf
    closest_L_idx  = -1
    closest_R_dist = Inf
    closest_R_idx  = -1
    
    # Check neighbors of i
    start_ptr = pg.neighbor_pointers[i]
    for n in 0:(pg.num_neighbors[i]-1)
        flat_idx = start_ptr + n
        dist_from_i = pg.neighbor_xdistance[flat_idx]
        nb_idx      = pg.neighbor_indices[flat_idx]
        
        # Distance from new particle
        dist_from_new = dist_from_i - rel_pos
        
        if dist_from_new < 0 && dist_from_new > closest_L_dist
            closest_L_dist = dist_from_new
            closest_L_idx  = nb_idx
        elseif dist_from_new > 0 && dist_from_new < closest_R_dist
            closest_R_dist = dist_from_new
            closest_R_idx  = nb_idx
        end
    end
    
    # Also check 'i' itself
    dist_from_new_i = 0.0 - rel_pos
    if dist_from_new_i < 0 && dist_from_new_i > closest_L_dist
        closest_L_dist = dist_from_new_i
        closest_L_idx  = i
    elseif dist_from_new_i > 0 && dist_from_new_i < closest_R_dist
        closest_R_dist = dist_from_new_i
        closest_R_idx  = i
    end

    # 3. Calculate Rho
    new_rho = 0.0
    
    if closest_L_idx != -1 && closest_R_idx != -1
        # Case A: Interior Voxel (Bounded) -> Linear Interpolation
        rho_L = pg.rhos[closest_L_idx]
        rho_R = pg.rhos[closest_R_idx]
        
        total_dist = closest_R_dist - closest_L_dist
        factor = (0.0 - closest_L_dist) / total_dist
        new_rho = rho_L * (1.0 - factor) + rho_R * factor
        
    elseif closest_L_idx == -1 && closest_R_idx != -1
        # Case B: Left Boundary (Void on Left) -> Extrapolate from Right
        # We have P_Right. We need P_Right's neighbor to the Right (P_RR) for gradient.
        
        P_R = closest_R_idx
        P_RR = _find_nearest_neighbor_in_dir(pg, P_R, 1) # Dir 1 = Right
        
        if P_RR != -1
            # Linear Extrapolation
            # Slope = (rho_RR - rho_R) / (pos_RR - pos_R)
            # rho_new = rho_R + Slope * (pos_new - pos_R)
            
            rho_R  = pg.rhos[P_R]
            rho_RR = pg.rhos[P_RR]
            
            # Distances (be careful with periodicity)
            dist_R_to_RR = getDistance(pg, P_R, P_RR)
            dist_R_to_New = -closest_R_dist # Negative because New is left of R
            
            if abs(dist_R_to_RR) > 1e-9
                slope = (rho_RR - rho_R) / dist_R_to_RR
                new_rho = rho_R + slope * dist_R_to_New
            else
                new_rho = rho_R
            end
        else
            # Fallback: Constant
            new_rho = pg.rhos[P_R]
        end
        
    elseif closest_L_idx != -1 && closest_R_idx == -1
        # Case C: Right Boundary (Void on Right) -> Extrapolate from Left
        # We have P_Left. We need P_Left's neighbor to the Left (P_LL).
        
        P_L = closest_L_idx
        P_LL = _find_nearest_neighbor_in_dir(pg, P_L, -1) # Dir -1 = Left
        
        if P_LL != -1
            rho_L  = pg.rhos[P_L]
            rho_LL = pg.rhos[P_LL]
            
            dist_L_to_LL = getDistance(pg, P_L, P_LL)
            dist_L_to_New = -closest_L_dist # Positive
            
            if abs(dist_L_to_LL) > 1e-9
                slope = (rho_LL - rho_L) / dist_L_to_LL
                new_rho = rho_L + slope * dist_L_to_New
            else
                new_rho = rho_L
            end
        else
            new_rho = pg.rhos[P_L]
        end
    else
        # Case D: Isolated (Shouldn't happen with min_nb > 0 check)
        return 
    end
    
    # 4. Limit/Clamp Rho (Optional stability)
    # new_rho = max(0.0, min(1.0, new_rho)) # If you know bounds
    
    push!(pg.split_buffer_pos, abs_pos)
    push!(pg.split_buffer_rho, new_rho)
end

"""
    _find_nearest_neighbor_in_dir(pg, idx, dir)

Finds the nearest neighbor of `idx` in the specified direction 
(-1 for Left, 1 for Right). Returns -1 if none found.
"""
function _find_nearest_neighbor_in_dir(pg, idx, dir)
    best_dist = Inf
    best_idx = -1
    
    start_ptr = pg.neighbor_pointers[idx]
    for n in 0:(pg.num_neighbors[idx]-1)
        flat_idx = start_ptr + n
        dist = pg.neighbor_xdistance[flat_idx]
        
        if dir > 0 # Search Right (Positive Dist)
            if dist > 0 && dist < best_dist
                best_dist = dist
                best_idx = pg.neighbor_indices[flat_idx]
            end
        else # Search Left (Negative Dist)
            if dist < 0 && abs(dist) < best_dist
                best_dist = abs(dist)
                best_idx = pg.neighbor_indices[flat_idx]
            end
        end
    end
    return best_idx
end
# Enums for clarity (or just use Ints)
# 0: Neighbor
# 1: Self
# 2: Horizon

# """
#     check_and_split_particle!(pg::ParticleGrid1D, i::Int)

# Asymmetric Splitting Logic:
# 1. Left Check: ONLY checks for gaps between the Left Horizon (-R) and Self. 
#    Ignores left neighbors (because those neighbors will handle the gap from their 'Right' side).
# 2. Right Check: Checks for gaps between Self and Neighbors, and Self and Right Horizon.
# """
# function check_and_split_particle!(pg::ParticleGrid1D, i::Int)
#     R = pg.max_dist
#     rho_i = pg.rhos[i]

#     # --- 1. LEFT SIDE (Boundary Only) ---
#     # We only care if we are touching the Left Horizon.
#     # We do NOT collect neighbors here.
    
#     # Check if we have ANY neighbors on the left.
#     # If we have a left neighbor, we assume THEY handle the gap between us.
#     # If we have NO left neighbors (or too few to cover the distance), we might need to fill the void to the horizon.
    
#     start_ptr = pg.neighbor_pointers[i]
#     num_nbs   = pg.num_neighbors[i]
    
#     has_left_neighbor = false
    
#     # Scan for left neighbors
#     for k in 0:(num_nbs - 1)
#         flat_idx = start_ptr + k 
#         dist = pg.neighbor_xdistance[flat_idx]
#         if dist < 0 && dist > -R
#             has_left_neighbor = true
#             break 
#         end
#     end
    
#     # If we have no left neighbors within R, we are exposed to the Left Horizon.
#     # We must fill this gap ourselves.
#     if !has_left_neighbor
#         # We simulate a "gap" between Left Horizon (-R) and Self (0.0)
#         # We verify if this gap needs splitting based on density requirements.
#         # Since count is 0, it's definitely < min_nb.
        
#         # Create a virtual buffer for the boundary routine
#         # (-R, type=Horizon), (0.0, type=Self)
#         left_points = [(-R, rho_i, 2), (0.0, rho_i, 1)]
#         _process_side_split!(pg, i, left_points)
#     end

#     # --- 2. RIGHT SIDE (Neighbors + Horizon) ---
#     # We handle ALL splitting to our right.
    
#     # Structure: (dist, rho, type_id) -> 0=Neighbor, 1=Self, 2=Horizon
#     right_points = sizehint!(Vector{Tuple{Float64, Float64, Int}}(), pg.max_nb + 2)
    
#     # Add Self
#     push!(right_points, (0.0, rho_i, 1))
    
#     # Add Right Neighbors Only
#     for k in 0:(num_nbs - 1)
#         flat_idx = start_ptr + k 
#         dist = pg.neighbor_xdistance[flat_idx]
        
#         if dist > 0 && dist < R
#             nb_idx = pg.neighbor_indices[flat_idx]
#             nb_rho = pg.rhos[nb_idx]
#             push!(right_points, (dist, nb_rho, 0))
#         end
#     end

#     # Add Right Horizon
#     push!(right_points, (R, rho_i, 2))
    
#     # Sort
#     sort!(right_points, by = x -> x[1])
    
#     # Check density: (Total points - Self - Horizon)
#     neighbor_count = length(right_points) - 2
    
#     if neighbor_count < pg.min_nb
#         _process_side_split!(pg, i, right_points)
#     end
# end

# """
#     _process_side_split!(pg, i, points)

# Finds the largest gap and interpolates.
# """
# function _process_side_split!(pg::ParticleGrid1D, i::Int, points::Vector{Tuple{Float64, Float64, Int}})
#     max_gap = -1.0
#     gap_idx = -1
    
#     # Find Largest Gap
#     for k in 1:(length(points) - 1)
#         gap = points[k+1][1] - points[k][1]
#         if gap > max_gap
#             max_gap = gap
#             gap_idx = k
#         end
#     end
    
#     if gap_idx != -1
#         p1 = points[gap_idx]
#         p2 = points[gap_idx+1]
        
#         # Interpolate Position
#         new_rel_dist = (p1[1] + p2[1]) / 2.0
#         new_abs_pos  = pg.positions[i] + new_rel_dist
        
#         # --- Boundary & Domain Check ---
#         if pg.bc == :periodic
#             domain = pg.xmax - pg.xmin
#             if new_abs_pos > pg.xmax; new_abs_pos -= domain; end
#             if new_abs_pos < pg.xmin; new_abs_pos += domain; end
#         else
#             # STRICT DOMAIN CLIPPING
#             if new_abs_pos < pg.xmin || new_abs_pos > pg.xmax
#                 return 
#             end
#         end

#         # Interpolate Density
#         new_rho = 0.0
        
#         # 1. Horizon -> Self (Left Boundary Case)
#         # p1 is Horizon(-R), p2 is Self(0.0)
#         if p1[3] == 2 && p2[3] == 1 
#              new_rho = p2[2] # Constant Extrapolation from Self
        
#         # 2. Self -> Horizon (Right Boundary Case)
#         # p1 is Self(0.0), p2 is Horizon(R)
#         elseif p1[3] == 1 && p2[3] == 2
#              new_rho = p1[2] # Constant Extrapolation from Self
             
#         # 3. Neighbor -> Horizon (Right Edge of Neighbor group)
#         elseif p1[3] == 0 && p2[3] == 2
#              new_rho = p1[2] # Constant Extrapolation from Neighbor

#         # 4. Standard: Self -> Neighbor OR Neighbor -> Neighbor
#         else
#              new_rho = 0.5 * (p1[2] + p2[2]) # Linear
#         end
        
#         push!(pg.split_buffer_pos, new_abs_pos)
#         push!(pg.split_buffer_rho, new_rho)
#     end
# end


# function manage_particles!(pg::ParticleGrid1D)
#     # 1. Reset Buffers
#     # We only need to clear the active region of the merged buffer
#     # But fill! is fast enough and safe.
#     if length(pg.merged_buffer) < pg.N
#         safe_resize!(pg.merged_buffer, pg.N)
#     end
#     fill!(view(pg.merged_buffer, 1:pg.N), false)
    
#     empty!(pg.split_buffer_pos)
#     empty!(pg.split_buffer_rho)
    
#     # Accessors for speed
#     pos = pg.positions
#     rhos = pg.rhos
#     vols = pg.volumes
#     is_bd = pg.is_boundary
#     merged = pg.merged_buffer
    
#     # Neighbor accessors
#     nb_indices = pg.neighbor_indices
#     nb_dists   = pg.neighbor_xdistance
#     nb_ptrs    = pg.neighbor_pointers
#     num_nbs    = pg.num_neighbors
    
#     write_idx = 0 # The "In-Situ" Pointer

#     # --- MAIN COMPACTION LOOP ---
#     # We iterate only up to the current active N
#     for i in 1:pg.N
#         if merged[i]; continue; end

#         write_idx += 1
        
#         # Accumulators
#         x = pos[i]
#         rho = rhos[i]
        
#         boundary_votes = is_bd[i] ? 1 : 0
#         total_votes    = 1
        
#         # Check Neighbors
#         start_ptr = nb_ptrs[i]
#         n_count   = num_nbs[i]
#         n_count = 0
#         if n_count > 0
#             end_ptr = start_ptr + n_count - 1
#             for k in start_ptr:end_ptr
#                 j = nb_indices[k]
                
#                 # Merge with future particles (j > i) to avoid double processing
#                 if j > i && !merged[j]
#                     dist = abs(nb_dists[k]) 
                    
#                     if dist < pg.min_dist
#                         # --- MERGE ---
#                         x += pos[j]
#                         rho += rhos[j]
                        
#                         if is_bd[j]; boundary_votes += 1; end
#                         total_votes += 1
                        
#                         merged[j] = true
#                     end
#                 end
#             end
#         end
#         normalization = total_votes
#         # Write compacted result
#         pos[write_idx]  = x / normalization
#         rhos[write_idx] = rho / normalization
        
#         #is_bd[write_idx] = (boundary_votes > total_votes / 2)
#         check_and_split_particle!(pg,i)
#         # Note: We do NOT split here. We wait until the grid is sorted.
#     end
#     println("STEP DONE")
#     # --- 2. Finalize Grid Structure ---
    
#     # Number of particles surviving the merge
#     N_merged = write_idx
    
#     # Number of new particles to add
#     N_new = length(pg.split_buffer_pos)
#     N_total = N_merged + N_new
    
#     # Ensure Capacity for ALL persistent fields
#     safe_resize!(pg.positions, N_total)
#     safe_resize!(pg.rhos, N_total)
#     safe_resize!(pg.curvatures, N_total)
#     safe_resize!(pg.is_boundary, N_total)
#     safe_resize!(pg.volumes, N_total)
#     safe_resize!(pg.mood_events, N_total)

#     # --- Manual Append (Avoids Allocations) ---
#     if N_new > 0
#         # Copy split particles into the "tail" of the arrays
#         for k in 1:N_new
#             idx = N_merged + k
#             pg.positions[idx]   = pg.split_buffer_pos[k]
#             pg.rhos[idx]        = pg.split_buffer_rho[k]
            
#             # Initialize defaults
#             pg.curvatures[idx]  = 0.0
#             pg.is_boundary[idx] = false
#             pg.volumes[idx]     = 0.0 # Will be recalculated
#             pg.mood_events[idx] = false
#         end
#     end

#     # --- 3. Update Global State ---
#     pg.N = N_total
#     if pg.N > 2000; error("Too many particles!") end
#     # Update indices ranges
#     if pg.bc == :periodic
#          pg.interior_indices = 1:pg.N
#     else
#          pg.interior_indices = (pg.N_ghost + 1):(pg.N - pg.N_ghost)
#     end
    
#     # Resize auxiliary buffers for the NEW N (pointers is N+1)
#     safe_resize!(pg.num_neighbors, pg.N)
#     safe_resize!(pg.neighbor_pointers, pg.N + 1)
#     safe_resize!(pg.merged_buffer, pg.N)

#     # --- 4. Sort and Rebuild ---
#     # Essential for 1D logic: places appended particles into correct gaps
#     #sort_1d_particles!(pg)
    
#     updateNeighbors!(pg)
#     determineVolumes!(pg)
# end

# """
#     check_and_split_particle!(pg::ParticleGrid1D, i::Int)

# Checks if particle `i` has enough neighbors on the Left and Right sides.
# If `count < min_nb`, it identifies the largest gap on that side and inserts
# a new particle, using linear interpolation for particle-particle gaps
# and constant extrapolation for horizon-particle gaps.
# """
# function check_and_split_particle!(pg::ParticleGrid1D, i::Int)
#     # 1. Gather Neighborhood Data
#     # We need to sort neighbors into Left and Right lists to check coverage.
#     R = pg.max_dist
    
#     # Structure: (relative_distance, rho_value, is_horizon_marker)
#     # We use a flag `is_horizon_marker` to detect if a gap touches the empty horizon.
#     left_points  = sizehint!(Vector{Tuple{Float64, Float64, Bool}}(), pg.max_nb + 2)
#     right_points = sizehint!(Vector{Tuple{Float64, Float64, Bool}}(), pg.max_nb + 2)
    
#     # Add Self (at 0.0)
#     rho_i = pg.rhos[i]
#     push!(left_points,  (0.0, rho_i, false))
#     push!(right_points, (0.0, rho_i, false))
    
#     # Add Neighbors
#     start_ptr = pg.neighbor_pointers[i]
#     num_nbs   = pg.num_neighbors[i]
    
#     for k in 0:(num_nbs - 1)
#         # Note: In your SoA layout, neighbor indices are flattened. 
#         # Ensure your access here matches your specific implementation.
#         # Assuming: global index = start_ptr + k
#         flat_idx = start_ptr + k 
        
#         dist = pg.neighbor_xdistance[flat_idx]
#         w    = pg.neighbor_weights[flat_idx] # Or access rho via index if needed
        
#         # We need the neighbor's rho. 
#         # If your neighbor list doesn't store rho, look it up:
#         nb_idx = pg.neighbor_indices[flat_idx]
#         nb_rho = pg.rhos[nb_idx]

#         if dist < 0 && dist > -R
#             push!(left_points, (dist, nb_rho, false))
#         elseif dist > 0 && dist < R
#             push!(right_points, (dist, nb_rho, false))
#         end
#     end

#     # 2. Process Left Side (Interval [-R, 0])
#     # Add Horizon Marker at -R
#     # We use rho_i as a placeholder; the logic handles the constant interp.
#     push!(left_points, (-R, rho_i, true)) 
#     sort!(left_points, by = x -> x[1])
    
#     # Count real neighbors (Total points - Self - Horizon)
#     left_count = length(left_points) - 2
#     if left_count < pg.min_nb
#         _process_side_split!(pg, i, left_points)
#     end

#     # 3. Process Right Side (Interval [0, R])
#     push!(right_points, (R, rho_i, true))
#     sort!(right_points, by = x -> x[1])
    
#     right_count = length(right_points) - 2
#     if right_count < pg.min_nb
#         _process_side_split!(pg, i, right_points)
#     end
#     print(left_count,right_count)
# end

# """
#     _process_side_split!(pg, i, points)

# Helper to find the largest gap in a sorted list of (dist, rho, is_horizon) 
# and insert a split particle.
# """
# function _process_side_split!(pg::ParticleGrid1D, i::Int, points::Vector{Tuple{Float64, Float64, Bool}})
#     max_gap = -1.0
#     gap_idx = -1
    
#     # 1. Find Largest Gap
#     for k in 1:(length(points) - 1)
#         curr = points[k]
#         next = points[k+1]
        
#         gap = next[1] - curr[1]
#         if gap > max_gap
#             max_gap = gap
#             gap_idx = k
#         end
#     end
    
#     # 2. Interpolate Logic
#     if gap_idx != -1
#         p1 = points[gap_idx]
#         p2 = points[gap_idx+1]
        
#         # Position: Always midpoint of the gap
#         new_rel_dist = (p1[1] + p2[1]) / 2.0
        
#         # Density: Constant vs Linear
#         # If either side of the gap is the "Horizon Marker", use Constant.
#         # Otherwise (Particle-to-Particle), use Linear.
#         new_rho = 0.0
        
#         if p1[3] # p1 is Horizon (Left edge case: Horizon -> Particle)
#             # Constant extrapolation from the inner particle (p2)
#             new_rho = p2[2]
#         elseif p2[3] # p2 is Horizon (Right edge case: Particle -> Horizon)
#             # Constant extrapolation from the inner particle (p1)
#             new_rho = p1[2]
#         else
#             # Standard Linear Interpolation
#             new_rho = 0.5 * (p1[2] + p2[2])
#         end
        
#         # 3. Calculate Global Position & Wrap
#         new_abs_pos = pg.positions[i] + new_rel_dist
        
#         if pg.bc == :periodic
#             domain = pg.xmax - pg.xmin
#             if new_abs_pos > pg.xmax; new_abs_pos -= domain; end
#             if new_abs_pos < pg.xmin; new_abs_pos += domain; end
#         end
        
#         # 4. Push to Split Buffer
#         push!(pg.split_buffer_pos, new_abs_pos)
#         push!(pg.split_buffer_rho, new_rho)
#     end
# end

# """
#     fill_gaps_on_side!(pg, points_buffer, i, min_nb, R)

# Helper to iterate over a sorted buffer of (relative_dist, density) points
# representing ONE side of the neighborhood (e.g., [-R, 0]).
# Splits gaps until `min_nb` neighbors exist or gaps are sufficiently small.
# """
# function fill_gaps_on_side!(pg::ParticleGrid1D, points_buffer::Vector{Tuple{Float64, Float64}}, i::Int, min_nb::Real, R::Float64)
#     # Count valid neighbors (Total points minus the two anchors: Self and Horizon)
#     # points_buffer contains: [Horizon, ...Neighbors..., Self] (or vice versa)
#     current_nb = length(points_buffer) - 2
    
#     # Iterate until requirements are met
#     # We add a safety break to prevent infinite loops in degenerate cases
#     max_iter = 10
#     iter = 0
    
#     while current_nb < min_nb && iter < max_iter
#         iter += 1
        
#         # 1. Find the Largest Gap within this side
#         max_gap = -1.0
#         gap_idx = -1
        
#         for k in 1:(length(points_buffer)-1)
#             # Gap between k and k+1
#             gap = abs(points_buffer[k+1][1] - points_buffer[k][1])
#             if gap > max_gap
#                 max_gap = gap
#                 gap_idx = k
#             end
#         end
        
#         # 2. Check Stopping Conditions
#         # If we have enough neighbors AND the gap is within the resolution limit (R), stop.
#         # Note: If min_nb is not met, we continue splitting even if gap < R.
#         if current_nb >= min_nb && max_gap <= R
#             break
#         end

#         # 3. Split the Gap
#         p_1 = points_buffer[gap_idx]
#         p_2 = points_buffer[gap_idx+1]
        
#         # Position: Midpoint
#         new_rel_pos = (p_1[1] + p_2[1]) / 2.0
        
#         # Density: Linear Interpolation
#         # (Preserves gradients, prevents zig-zags)
#         new_rho = 0.5 * (p_1[2] + p_2[2])
        
#         # --- Add to Global Grid Buffer ---
#         new_abs_pos = pg.positions[i] + new_rel_pos
        
#         # Periodic Wrap
#         if pg.bc == :periodic
#             L = pg.xmax - pg.xmin
#             if new_abs_pos > pg.xmax; new_abs_pos -= L; end
#             if new_abs_pos < pg.xmin; new_abs_pos += L; end
#         end
        
#         push!(pg.split_buffer_pos, new_abs_pos)
#         push!(pg.split_buffer_rho, new_rho)
        
#         # --- Update Local Buffer ---
#         # Insert the new point to potentially split it again if min_nb > 1
#         insert!(points_buffer, gap_idx + 1, (new_rel_pos, new_rho))
#         current_nb += 1
#     end
# end

# function check_and_split_particle!(pg::ParticleGrid1D, i::Int, n_count::Int, start_ptr::Int)
#     # Horizon Radius
#     R = pg.max_dist
    
#     # --- Prepare Buffers for ONE-SIDED Checks ---
#     # We use the current particle 'i' as the anchor (0.0).
#     rho_i = pg.rhos[i]
    
#     # LEFT Side: [-R, 0]
#     # Always include Horizon (-R) and Self (0.0)
#     # We use 'rho_i' for the horizon density to ensure flat extrapolation into voids.
#     left_points = Vector{Tuple{Float64, Float64}}()
#     sizehint!(left_points, n_count + 2)
#     push!(left_points, (-R, rho_i))
#     push!(left_points, (0.0, rho_i))
    
#     # RIGHT Side: [0, R]
#     right_points = Vector{Tuple{Float64, Float64}}()
#     sizehint!(right_points, n_count + 2)
#     push!(right_points, (0.0, rho_i))
#     push!(right_points, (R, rho_i))

#     # --- Populate from Neighbors ---
#     if n_count > 0
#         end_ptr = start_ptr + n_count - 1
#         for k in start_ptr:end_ptr
#             dist = pg.neighbor_xdistance[k]
            
#             # Check range and classify
#             if dist > -R && dist < 0.0
#                 # LEFT Neighbor
#                 j = pg.neighbor_indices[k]
#                 push!(left_points, (dist, pg.rhos[j]))
                
#             elseif dist > 0.0 && dist < R
#                 # RIGHT Neighbor
#                 j = pg.neighbor_indices[k]
#                 push!(right_points, (dist, pg.rhos[j]))
#             end
#         end
#     end
    
#     # --- Sort Buffers ---
#     # Left: [-R, ..., 0]
#     sort!(left_points, by = x -> x[1])
#     # Right: [0, ..., R]
#     sort!(right_points, by = x -> x[1])

#     # --- Execute One-Sided Logic ---
#     # We require 'min_nb' neighbors on EACH side independently.
    
#     fill_gaps_on_side!(pg, left_points, i, pg.min_nb, R)
#     fill_gaps_on_side!(pg, right_points, i, pg.min_nb, R)
# end

# function manage_particles!(pg::ParticleGrid1D)
#     # --- 1. MERGE PHASE ---
#     # Merge particles that are too clumpy (d < min_dist)
#     _merge_particles!(pg)
    
#     # --- 2. SORT PHASE ---
#     # Crucial: We must sort to strictly define "neighbors" in 1D for splitting.
#     # This recovers the grid topology after particles have moved/merged.
#     sort_1d_particles!(pg)
    
#     # --- 3. SPLIT PHASE ---
#     # Fill gaps that are too large (d > max_split_dist)
#     # We use a threshold relative to the nominal spacing dx.
#     # Typically 1.3 to 1.5 times dx prevents holes without over-refining.
#     max_split_dist = pg.dx * 1.2
#     #_fill_gaps_1d!(pg)

#     # --- 4. FINALIZE ---
#     # Rebuild neighbor lists now that N and positions have changed
#     updateNeighbors!(pg)
#     determineVolumes!(pg)
# end

# """
#     _merge_particles!(pg::ParticleGrid1D)

# Identifies particles closer than `pg.min_dist` and merges them into a single 
# particle (conserving volume-weighted moments). Compacts the arrays in-place.
# """
# function _merge_particles!(pg::ParticleGrid1D)
#     # Reset merge buffer
#     if length(pg.merged_buffer) < pg.N
#         safe_resize!(pg.merged_buffer, pg.N)
#     end
#     fill!(view(pg.merged_buffer, 1:pg.N), false)

#     # Accessors
#     pos    = pg.positions
#     rhos   = pg.rhos
#     vols   = pg.volumes
#     is_bd  = pg.is_boundary
#     merged = pg.merged_buffer
    
#     # Neighbor accessors
#     nb_indices = pg.neighbor_indices
#     nb_dists   = pg.neighbor_xdistance
#     nb_ptrs    = pg.neighbor_pointers
#     num_nbs    = pg.num_neighbors

#     write_idx = 0 

#     for i in 1:pg.N
#         if merged[i]; continue; end

#         write_idx += 1
        
#         # Accumulators
#         x = pos[i]
#         rho = rhos[i]
        
#         boundary_votes = is_bd[i] ? 1 : 0
#         total_votes    = 1
        
#         # Check Neighbors
#         start_ptr = nb_ptrs[i]
#         n_count   = num_nbs[i]
        
#         if n_count > 0
#             end_ptr = start_ptr + n_count - 1
#             for k in start_ptr:end_ptr
#                 j = nb_indices[k]
                
#                 # Merge with future particles (j > i) to avoid double processing
#                 if j > i && !merged[j]
#                     dist = abs(nb_dists[k]) 
                    
#                     if dist < pg.min_dist
#                         # --- MERGE ---
#                         x += pos[j]
#                         rho += rhos[j]
                        
#                         if is_bd[j]; boundary_votes += 1; end
#                         total_votes += 1
                        
#                         merged[j] = true
#                     end
#                 end
#             end
#         end
#         normalization = total_votes
#         # Write compacted result
#         pos[write_idx]  = x / normalization
#         rhos[write_idx] = rho / normalization
        
#         is_bd[write_idx] = (boundary_votes > total_votes / 2)
#         check_and_split_particle!(pg,i)
#         # Note: We do NOT split here. We wait until the grid is sorted.
#     end

#     # Update N to the new compacted size
#     pg.N = write_idx
    
#     # Update indices ranges
#     if pg.bc == :periodic
#          pg.interior_indices = 1:pg.N
#     else
#          pg.interior_indices = (pg.N_ghost + 1):(pg.N - pg.N_ghost)
#     end

#     return nothing
# end

