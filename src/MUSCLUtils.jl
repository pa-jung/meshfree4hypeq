# --- In MUSCL.jl ---

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
(Order 1) No higher derivatives to calculate.
"""
@inline function _calculate_higher_derivatives(::MUSCLORDER1, nb_slice, df_neighbors, ws)
    return () # Return empty tuple
end

"""
(Order 2) Calculate curve_xx, curve_yy, curve_xy.
"""
@inline function _calculate_higher_derivatives(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    df_neighbors::AbstractVector,
    ws::MUSCLWorkspace2D2O # Requires the O2 workspace
)
    curve_xx = 0.0
    curve_yy = 0.0
    curve_xy = 0.0

    # O2 higher derivatives use alfaij_bar, betaij_bar, gammaij
    @inbounds for k in nb_slice
        df = df_neighbors[k]
        curve_xx += ws.alfaij_bars[k] * df
        curve_yy += ws.betaij_bars[k] * df
        curve_xy += ws.gammaijs[k] * df
    end
    return curve_xx, curve_yy, curve_xy
end

# --- In MUSCL.jl ---

"""
(Order 1) Save only slopes.
"""
@inline function _save_derivatives!(
    ws::MUSCLWorkspace2D1O, # Dispatches on O1 workspace
    i::Int,
    slope_x::Real, slope_y::Real,
    higher_derivatives::Tuple{} # Expects empty tuple
)
    ws.slopes_x[i] = slope_x
    ws.slopes_y[i] = slope_y
end

"""
(Order 2) Save slopes and curves.
"""
@inline function _save_derivatives!(
    ws::MUSCLWorkspace2D2O, # Dispatches on O2 workspace
    i::Int,
    slope_x::Real, slope_y::Real,
    higher_derivatives::NTuple{3, Real} # Expects (cxx, cyy, cxy)
)
    ws.slopes_x[i]  = slope_x
    ws.slopes_y[i]  = slope_y
    ws.curves_xx[i] = higher_derivatives[1]
    ws.curves_yy[i] = higher_derivatives[2]
    ws.curves_xy[i] = higher_derivatives[3]
end

# --- Reconstruction Helpers for fij and fji (2D) ---
# These are kept as they are called by the new functor
# It uses pre-calculated slopes.
function reconstruct_interface_states(::MUSCLORDER1, ws::MUSCLWorkspace2D1O, fi, fj, p_idx, nb_idx, deltaX, deltaY)
    fij = fi  + 0.5 * (deltaX * ws.slopes_x[p_idx]  + deltaY * ws.slopes_y[p_idx])
    fji = fj - 0.5 * (deltaX * ws.slopes_x[nb_idx] + deltaY * ws.slopes_y[nb_idx])
    return fij, fji
end


# --- REFACTORED: Order 2 now just fetches pre-calculated values ---
function reconstruct_interface_states(::MUSCLORDER2, ws::MUSCLWorkspace2D2O, f_i, f_j, p_idx, nb_idx, deltaX, deltaY)
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