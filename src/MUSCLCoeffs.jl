# --- In MUSCL.jl, replace all old 1D _muscl_math_... helpers ---

@inline function _zero_coeffs!(nb_slice::UnitRange{Int}, ws::MUSCLWorkspace2D1O)
    @inbounds for k in nb_slice
        ws.alfaijs[k] = 0.0
        ws.betaijs[k] = 0.0
    end
end
@inline function _zero_coeffs!(nb_slice::UnitRange{Int}, ws::MUSCLWorkspace2D2O)
    @inbounds for k in nb_slice
        ws.alfaijs[k] = 0.0
        ws.betaijs[k] = 0.0
        ws.alfaij_bars[k] = 0.0
        ws.betaij_bars[k] = 0.0
        ws.gammaijs[k] = 0.0
    end
end

# --- ORDER 1 ---
function _compute_coeffs!(
    ::MUSCLORDER1,
    nb_slice::UnitRange{Int},
    dx::AbstractVector, w::AbstractVector, L::Float64,
    _a, alfaij_bar::AbstractVector, _b, _g # O1 only needs alfaij_bar
)
    invL = 1.0 / L
    
    # 1. Build 1x1 Normal Matrix N = A^T W A
    # Basis: p1 = dx/L
    N11 = 0.0
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_s = dx[k] * invL # dx_scaled
        N11 += w_k * dx_s * dx_s
    end
    
    if N11 < 1e-14
        @inbounds for k in nb_slice; alfaij_bar[k] = 0.0; end
        return
    end
    
    invN11 = 1.0 / N11

    # 2. Solve N*c = b for each neighbor
    @inbounds for k in nb_slice
        # RHS b_k = (A^T W)_k = p1_k * w_k
        b1 = (dx[k] * invL) * w[k]
        
        # Solve
        c1_scaled = invN11 * b1
        
        # Unscale and store: c1_unscaled = c1_scaled / L
        alfaij_bar[k] = c1_scaled * invL
    end
end

# --- ORDER 2 ---
function _compute_coeffs!(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    dx::AbstractVector, w::AbstractVector, L::Float64,
    _a, alfaij_bar::AbstractVector, betaij::AbstractVector, _g
)
    invL = 1.0 / L
    invL2 = invL * invL

    # 1. Build 2x2 Normal Matrix N = A^T W A
    # Basis: p1 = dx/L, p2 = 0.5 * (dx/L)^2
    N11 = 0.0; N12 = 0.0; N22 = 0.0
    
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_s = dx[k] * invL
        
        p1 = dx_s
        p2 = 0.5 * dx_s * dx_s

        N11 += w_k * p1 * p1
        N12 += w_k * p1 * p2
        N22 += w_k * p2 * p2
    end
    
    # 2. Hardcoded 2x2 Cholesky Decomposition (N = LL^T)
    l11_sq = N11
    if l11_sq < 1e-14
        _zero_coeffs_1d!(nb_slice, _a, alfaij_bar, betaij, _g)
        return
    end
    l11 = sqrt(l11_sq)
    inv_l11 = 1.0 / l11
    
    l21 = N12 * inv_l11
    
    l22_sq = N22 - l21*l21
    if l22_sq < 1e-14
        _zero_coeffs_1d!(nb_slice, _a, alfaij_bar, betaij, _g)
        return
    end
    l22 = sqrt(l22_sq)
    inv_l22 = 1.0 / l22

    # 3. Solve N*c = b for each neighbor
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_s = dx[k] * invL

        # RHS b_k = (A^T W)_k
        p1_k = dx_s
        p2_k = 0.5 * dx_s * dx_s
        b1 = p1_k * w_k
        b2 = p2_k * w_k

        # a) Forward Sub (Ly = b)
        y1 = b1 * inv_l11
        y2 = (b2 - l21*y1) * inv_l22

        # b) Backward Sub (L^T c = y)
        c2_scaled = y2 * inv_l22
        c1_scaled = (y1 - l21*c2_scaled) * inv_l11
        
        # Unscale and store
        alfaij_bar[k] = c1_scaled * invL  # c1 / L
        betaij[k]     = c2_scaled * invL2 # c2 / L^2
    end
end

# --- ORDER 3 ---
function _compute_coeffs!(
    ::MUSCLORDER3,
    nb_slice::UnitRange{Int},
    dx::AbstractVector, w::AbstractVector, L::Float64,
    alfaij::AbstractVector, alfaij_bar::AbstractVector, betaij::AbstractVector, _g
)
    invL = 1.0 / L
    invL2 = invL * invL
    invL3 = invL2 * invL

    # 1. Build 3x3 Normal Matrix N = A^T W A
    # Basis: p1 = dx/L, p2 = 0.5 * (dx/L)^2, p3 = (1/6) * (dx/L)^3
    N11 = 0.0; N12 = 0.0; N13 = 0.0
    N22 = 0.0; N23 = 0.0
    N33 = 0.0
    
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_s = dx[k] * invL
        
        p1 = dx_s
        p2 = 0.5 * dx_s * dx_s
        p3 = (1.0/6.0) * dx_s * dx_s * dx_s

        N11 += w_k*p1*p1; N12 += w_k*p1*p2; N13 += w_k*p1*p3
        N22 += w_k*p2*p2; N23 += w_k*p2*p3
        N33 += w_k*p3*p3
    end
    
    # 2. Hardcoded 3x3 Cholesky Decomposition (N = LL^T)
    l11_sq = N11
    if l11_sq < 1e-14; _zero_coeffs_1d!(nb_slice, alfaij, alfaij_bar, betaij, _g); return; end
    l11 = sqrt(l11_sq); inv_l11 = 1.0 / l11
    l21 = N12 * inv_l11
    l31 = N13 * inv_l11

    l22_sq = N22 - l21*l21
    if l22_sq < 1e-14; _zero_coeffs_1d!(nb_slice, alfaij, alfaij_bar, betaij, _g); return; end
    l22 = sqrt(l22_sq); inv_l22 = 1.0 / l22
    l32 = (N23 - l31*l21) * inv_l22
    
    l33_sq = N33 - l31*l31 - l32*l32
    if l33_sq < 1e-14; _zero_coeffs_1d!(nb_slice, alfaij, alfaij_bar, betaij, _g); return; end
    l33 = sqrt(l33_sq); inv_l33 = 1.0 / l33

    # 3. Solve N*c = b for each neighbor
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_s = dx[k] * invL

        # RHS b_k = (A^T W)_k
        p1_k = dx_s
        p2_k = 0.5 * dx_s * dx_s
        p3_k = (1.0/6.0) * dx_s * dx_s * dx_s
        b1 = p1_k * w_k
        b2 = p2_k * w_k
        b3 = p3_k * w_k

        # a) Forward Sub (Ly = b)
        y1 = b1 * inv_l11
        y2 = (b2 - l21*y1) * inv_l22
        y3 = (b3 - l31*y1 - l32*y2) * inv_l33

        # b) Backward Sub (L^T c = y)
        c3_scaled = y3 * inv_l33
        c2_scaled = (y2 - l32*c3_scaled) * inv_l22
        c1_scaled = (y1 - l21*c2_scaled - l31*c3_scaled) * inv_l11
        
        # Unscale and store (O3 div uses alfaij)
        alfaij_bar[k] = c1_scaled * invL  # c1 / L
        betaij[k]     = c2_scaled * invL2 # c2 / L^2
        alfaij[k]     = c3_scaled * invL3 # c3 / L^3
    end
end

# --- ORDER 4 ---
function _compute_coeffs!(
    ::MUSCLORDER4,
    nb_slice::UnitRange{Int},
    dx::AbstractVector, w::AbstractVector, L::Float64,
    alfaij::AbstractVector, alfaij_bar::AbstractVector, 
    betaij::AbstractVector, gammaij::AbstractVector
)
    invL = 1.0 / L
    invL2 = invL * invL
    invL3 = invL2 * invL
    invL4 = invL3 * invL

    # 1. Build 4x4 Normal Matrix N = A^T W A
    # Basis: p1 = dx/L, p2 = 0.5(dx/L)^2, p3 = (1/6)(dx/L)^3, p4 = (1/24)(dx/L)^4
    N11 = 0.0; N12 = 0.0; N13 = 0.0; N14 = 0.0
    N22 = 0.0; N23 = 0.0; N24 = 0.0
    N33 = 0.0; N34 = 0.0
    N44 = 0.0
    
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_s = dx[k] * invL
        dx_s2 = dx_s * dx_s
        
        p1 = dx_s
        p2 = 0.5 * dx_s2
        p3 = (1.0/6.0) * dx_s2 * dx_s
        p4 = (1.0/24.0) * dx_s2 * dx_s2

        N11 += w_k*p1*p1; N12 += w_k*p1*p2; N13 += w_k*p1*p3; N14 += w_k*p1*p4
        N22 += w_k*p2*p2; N23 += w_k*p2*p3; N24 += w_k*p2*p4
        N33 += w_k*p3*p3; N34 += w_k*p3*p4
        N44 += w_k*p4*p4
    end
    
    # 2. Hardcoded 4x4 Cholesky Decomposition (N = LL^T)
    l11_sq = N11
    if l11_sq < 1e-14; _zero_coeffs_1d!(nb_slice, alfaij, alfaij_bar, betaij, gammaij); return; end
    l11 = sqrt(l11_sq); inv_l11 = 1.0 / l11
    l21 = N12 * inv_l11
    l31 = N13 * inv_l11
    l41 = N14 * inv_l11

    l22_sq = N22 - l21*l21
    if l22_sq < 1e-14; _zero_coeffs_1d!(nb_slice, alfaij, alfaij_bar, betaij, gammaij); return; end
    l22 = sqrt(l22_sq); inv_l22 = 1.0 / l22
    l32 = (N23 - l31*l21) * inv_l22
    l42 = (N24 - l41*l21) * inv_l22
    
    l33_sq = N33 - l31*l31 - l32*l32
    if l33_sq < 1e-14; _zero_coeffs_1d!(nb_slice, alfaij, alfaij_bar, betaij, gammaij); return; end
    l33 = sqrt(l33_sq); inv_l33 = 1.0 / l33
    l43 = (N34 - l41*l31 - l42*l32) * inv_l33
    
    l44_sq = N44 - l41*l41 - l42*l42 - l43*l43
    if l44_sq < 1e-14; _zero_coeffs_1d!(nb_slice, alfaij, alfaij_bar, betaij, gammaij); return; end
    l44 = sqrt(l44_sq); inv_l44 = 1.0 / l44

    # 3. Solve N*c = b for each neighbor
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_s = dx[k] * invL
        dx_s2 = dx_s * dx_s

        # RHS b_k = (A^T W)_k
        p1_k = dx_s
        p2_k = 0.5 * dx_s2
        p3_k = (1.0/6.0) * dx_s2 * dx_s
        p4_k = (1.0/24.0) * dx_s2 * dx_s2
        b1 = p1_k * w_k
        b2 = p2_k * w_k
        b3 = p3_k * w_k
        b4 = p4_k * w_k

        # a) Forward Sub (Ly = b)
        y1 = b1 * inv_l11
        y2 = (b2 - l21*y1) * inv_l22
        y3 = (b3 - l31*y1 - l32*y2) * inv_l33
        y4 = (b4 - l41*y1 - l42*y2 - l43*y3) * inv_l44

        # b) Backward Sub (L^T c = y)
        c4_scaled = y4 * inv_l44
        c3_scaled = (y3 - l43*c4_scaled) * inv_l33
        c2_scaled = (y2 - l32*c3_scaled - l42*c4_scaled) * inv_l22
        c1_scaled = (y1 - l21*c2_scaled - l31*c3_scaled - l41*c4_scaled) * inv_l11
        
        # Unscale and store (O4 div uses... alfaij?)
        alfaij_bar[k] = c1_scaled * invL  # c1 / L
        betaij[k]     = c2_scaled * invL2 # c2 / L^2
        alfaij[k]     = c3_scaled * invL3 # c3 / L^3
        gammaij[k]    = c4_scaled * invL4 # c4 / L^4
    end
end

# --- In MUSCL.jl ---

# --- REPLACED _compute_coeffs! for 2D Order 1 ---
"""
(2D Order 1) Calculates coefficients (alfaij, betaij).
"""
function _compute_coeffs!(
    ::MUSCLORDER1,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace2D1O, # <-- Takes workspace
    dx::AbstractVector,
    dy::AbstractVector,
    w::AbstractVector
)
    # Get views into workspace coefficient arrays
    alfaij = ws.alfaijs
    betaij = ws.betaijs

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
    invD = 1.0 / D
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_k = dx[k]
        dy_k = dy[k]

        alfaij[k] = (w_k * (A22 * dx_k - A12 * dy_k)) * invD
        betaij[k] = (w_k * (A11 * dy_k - A12 * dx_k)) * invD
    end
end

# --- REPLACED _compute_coeffs! for 2D Order 2 ---
"""
(2D Order 2) Calculates coefficients using hard-coded Cholesky.
"""
function _compute_coeffs!(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace2D2O, # <-- Takes workspace
    dx::AbstractVector,
    dy::AbstractVector,
    w::AbstractVector
)
    # Get views into workspace coefficient arrays
    alfaij      = ws.alfaijs
    betaij      = ws.betaijs
    alfaij_bar  = ws.alfaij_bars
    betaij_bar  = ws.betaij_bars
    gammaij     = ws.gammaijs

    # --- 1. Build the 5x5 Normal Matrix N = A^T W A ---
    N11 = 0.0; N12 = 0.0; N13 = 0.0; N14 = 0.0; N15 = 0.0
    N22 = 0.0; N23 = 0.0; N24 = 0.0; N25 = 0.0
    N33 = 0.0; N34 = 0.0; N35 = 0.0
    N44 = 0.0; N45 = 0.0
    N55 = 0.0

    @inbounds for k in nb_slice
        w_k = w[k]
        if w_k == 0.0; continue; end

        dx_k = dx[k]
        dy_k = dy[k]

        # Basis functions
        b1 = dx_k
        b2 = dy_k
        b3 = 0.5 * dx_k^2
        b4 = 0.5 * dy_k^2
        b5 = dx_k * dy_k

        # Add contribution to upper triangle of N = A^T W A
        N11 += w_k * b1 * b1; N12 += w_k * b1 * b2; N13 += w_k * b1 * b3
        N14 += w_k * b1 * b4; N15 += w_k * b1 * b5
        N22 += w_k * b2 * b2; N23 += w_k * b2 * b3; N24 += w_k * b2 * b4
        N25 += w_k * b2 * b5
        N33 += w_k * b3 * b3; N34 += w_k * b3 * b4; N35 += w_k * b3 * b5
        N44 += w_k * b4 * b4; N45 += w_k * b4 * b5
        N55 += w_k * b5 * b5
    end

    # --- 2. Hardcoded Cholesky Decomposition (N = LLᵀ) ---
    l11_sq = N11
    if l11_sq < 1e-14; _zero_coeffs_2d!(nb_slice, ws); return; end
    l11 = sqrt(l11_sq); inv_l11 = 1.0 / l11
    l21 = N12 * inv_l11
    l31 = N13 * inv_l11
    l41 = N14 * inv_l11
    l51 = N15 * inv_l11

    l22_sq = N22 - l21*l21
    if l22_sq < 1e-14; _zero_coeffs_2d!(nb_slice, ws); return; end
    l22 = sqrt(l22_sq); inv_l22 = 1.0 / l22
    l32 = (N23 - l31*l21) * inv_l22
    l42 = (N24 - l41*l21) * inv_l22
    l52 = (N25 - l51*l21) * inv_l22

    l33_sq = N33 - l31*l31 - l32*l32
    if l33_sq < 1e-14; _zero_coeffs_2d!(nb_slice, ws); return; end
    l33 = sqrt(l33_sq); inv_l33 = 1.0 / l33
    l43 = (N34 - l41*l31 - l42*l32) * inv_l33
    l53 = (N35 - l51*l31 - l52*l32) * inv_l33

    l44_sq = N44 - l41*l41 - l42*l42 - l43*l43
    if l44_sq < 1e-14; _zero_coeffs_2d!(nb_slice, ws); return; end
    l44 = sqrt(l44_sq); inv_l44 = 1.0 / l44
    l54 = (N45 - l51*l41 - l52*l42 - l53*l43) * inv_l44

    l55_sq = N55 - l51*l51 - l52*l52 - l53*l53 - l54*l54
    if l55_sq < 1e-14; _zero_coeffs_2d!(nb_slice, ws); return; end
    l55 = sqrt(l55_sq); inv_l55 = 1.0 / l55

    # --- 3. Solve for Coefficients for EACH neighbor ---
    @inbounds for k in nb_slice
        w_k = w[k]

        # Build the RHS vector b_k = (A^T W)_k = A_k^T * w_k
        dx_k = dx[k]
        dy_k = dy[k]

        b1 = dx_k * w_k
        b2 = dy_k * w_k
        b3 = (0.5 * dx_k^2) * w_k
        b4 = (0.5 * dy_k^2) * w_k
        b5 = (dx_k * dy_k) * w_k

        # Solve N*c = b
        # a) Forward Substitution (Ly = b)
        y1 = b1 * inv_l11
        y2 = (b2 - l21*y1) * inv_l22
        y3 = (b3 - l31*y1 - l32*y2) * inv_l33
        y4 = (b4 - l41*y1 - l42*y2 - l43*y3) * inv_l44
        y5 = (b5 - l51*y1 - l52*y2 - l53*y3 - l54*y4) * inv_l55

        # b) Backward Substitution (Lᵀc = y)
        c5 = y5 * inv_l55
        c4 = (y4 - l54*c5) * inv_l44
        c3 = (y3 - l43*c4 - l53*c5) * inv_l33
        c2 = (y2 - l32*c3 - l42*c4 - l52*c5) * inv_l22
        c1 = (y1 - l21*c2 - l31*c3 - l41*c4 - l51*c5) * inv_l11

        # Store coefficients in the workspace views
        alfaij[k]     = c1
        betaij[k]     = c2
        alfaij_bar[k] = c3
        betaij_bar[k] = c4
        gammaij[k]    = c5
    end
end

"""
(2D Helper) Helper function to zero out 2D coefficients.
"""
@inline function _zero_coeffs_2d!(nb_slice::UnitRange{Int}, ws::MUSCLWorkspace2D)
    # This needs dispatch based on the concrete workspace type
    _zero_coeffs_2d!(nb_slice, ws)
end
@inline function _zero_coeffs_2d!(nb_slice::UnitRange{Int}, ws::MUSCLWorkspace2D1O)
    @inbounds for k in nb_slice
        ws.alfaijs[k] = 0.0
        ws.betaijs[k] = 0.0
    end
end
@inline function _zero_coeffs_2d!(nb_slice::UnitRange{Int}, ws::MUSCLWorkspace2D2O)
    @inbounds for k in nb_slice
        ws.alfaijs[k] = 0.0
        ws.betaijs[k] = 0.0
        ws.alfaij_bars[k] = 0.0
        ws.betaij_bars[k] = 0.0
        ws.gammaijs[k] = 0.0
    end
end