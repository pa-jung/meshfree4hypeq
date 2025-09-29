# ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  # Only relevant in 2D. In 1D, all algorithms are the same.
abstract type TiwariAlgorithm <: UpwindAlgorithm end  # Split domain in left and right for d/dx, and up and down for d/dy.
abstract type PraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type ClassicAlgorithm <: UpwindAlgorithm end  # Take all points 'behind' center point. 
abstract type RusanovAlgorithm <: UpwindAlgorithm end # This is no upwinding of course but easy implementation in this framework (numerical Flux given does not have to be upwind)

# Define a workspace to hold temporary arrays for Upwind calculations
struct UpwindWorkspace
    dxVec::Vector{Float64}
    dyVec::Vector{Float64}
    dfVec::Vector{Float64}
    wVec::Vector{Float64}
    # For Tiwari algorithm
    xWindow::BitVector
    yWindow::BitVector

    function UpwindWorkspace(max_neighbors::Int=20) # Preallocate with a reasonable capacity
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

# Helper to ensure workspace vectors are large enough
function ensure_capacity!(ws::UpwindWorkspace, n::Int)
    if length(ws.dxVec) < n
        resize!.((ws.dxVec, ws.dyVec, ws.dfVec, ws.wVec, ws.xWindow, ws.yWindow), n)
    end
end


struct UpwindGradient{Algorithm <: UpwindAlgorithm} <: GradientInterpolator
    order::Int
    res::Vector{Float64}
    weightFunction::MLSWeightFunction
    numericalFlux::NumericalFluxFunction
    workspace::UpwindWorkspace

    """
        UpwindGradient(order::Int64 = 1; algType::String = "")

    Constructor for Upwind Object. algType only has impact in 2D upwinding.
    """
    function UpwindGradient(order::Int=1; numericalFlux::NumericalFluxFunction=UpwindFlux(), algType::String="Classic", weightFunction::MLSWeightFunction=exponentialWeightFunction())
        @assert order >= 1 "Order must be larger or equal to one."
        @assert algType in ["Classic", "Tiwari", "Praveen", "NonLinearPraveen"]
        if order == 1
            size = 2  # In 2D res has length 2, in 1D res has length 1
        elseif order == 2
            size = 5  # In 2D res had length 5, in 1D res has length 2
        end
        local alg_type
        if algType == "Classic"
            alg_type = ClassicAlgorithm
        elseif algType == "Praveen"
            @assert order == 1
            alg_type = PraveenAlgorithm
        elseif algType == "NonLinearPraveen"
            @assert order == 1
            alg_type = NonLinearPraveenAlgorithm
        elseif algType == "Tiwari"
            alg_type = TiwariAlgorithm
        end
        new{alg_type}(order, Vector{Float64}(undef, size), weightFunction, numericalFlux, UpwindWorkspace())
    end
end

# ... (keep all other content in Interpolations.jl) ...

#==============================================================================
  UPWIND GRADIENT (Optimized for SoA Grids)
==============================================================================#

# --- REFACTORED 1D Upwind Functor ---
function (upwind::UpwindGradient)(
    particleGrid::ParticleGrid1D,
    particleIndex::Integer,
    fVec::AbstractVector{<:Real},
    eq::ScalarHyperbolicPDE,
    settings::SimSetting;
    setCurvature::Bool=true
)::Real
    
    neighbors = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbors)
    ws = upwind.workspace
    ensure_capacity!(ws, num_neighbors)

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
    
    wVec .= upwind.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec, wVec, dfVec, upwind.res; order=upwind.order)

    if setCurvature
        particleGrid.curvatures[particleIndex] = 0.0
    end
    
    return 2 * upwind.res[1] / settings.interpRange
end


# --- REFACTORED 2D Upwind Functor (Classic Algorithm) ---
function (upwind::UpwindGradient{ClassicAlgorithm})(
    particleGrid::ParticleGrid2D,
    particleIndex::Integer,
    fVec::AbstractVector{<:Real},
    eq::LinearAdvection{2},
    settings::SimSetting;
    setCurvature::Bool=true
)::Real
    
    vel = velocity(eq, 0.0)
    
    # Filter to get upwind neighbors first
    upwind_indices = [nb for nb in particleGrid.neighbour_indices[particleIndex] if dot(getDistance(particleGrid, particleIndex, nb), vel) < 0]
    num_upwind = length(upwind_indices)

    ws = upwind.workspace
    ensure_capacity!(ws, num_upwind)

    # Use views into the workspace
    dxVec = @view ws.dxVec[1:num_upwind]
    dyVec = @view ws.dyVec[1:num_upwind]
    dfVec = @view ws.dfVec[1:num_upwind]
    wVec = @view ws.wVec[1:num_upwind]

    for (i, nbIndex) in enumerate(upwind_indices)
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX / settings.interpRange
        dyVec[i] = deltaY / settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end

    wVec .= upwind.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec, dyVec, wVec, dfVec, upwind.res; order=upwind.order)

    if setCurvature && upwind.order == 2
        particleGrid.curvatures[particleIndex, 1] = upwind.res[3] / (settings.interpRange^2)
        particleGrid.curvatures[particleIndex, 2] = upwind.res[4] / (settings.interpRange^2)
    end
    
    return dot(vel, @view(upwind.res[1:2])) / settings.interpRange
end


# --- 2D Upwind Functor (Tiwari Algorithm) ---
function (upwind::UpwindGradient{TiwariAlgorithm})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector{<:Real}, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    vel = velocity(eq, 0.0) # Velocity is constant for LinearAdvection
    neighbour_indices = particleGrid.neighbour_indices[particleIndex]
    num_neighbours = length(neighbour_indices)

    # Temporary buffers
    dxVec = Vector{Float64}(undef, num_neighbours)
    dyVec = Vector{Float64}(undef, num_neighbours)
    dfVec = Vector{Float64}(undef, num_neighbours)    
    xWindow = falses(num_neighbours)
    yWindow = falses(num_neighbours)

    for (i, nbIndex) in enumerate(neighbour_indices)
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX / settings.interpRange
        dyVec[i] = deltaY / settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        xWindow[i] = ((vel[1] >= 0.0) && (deltaX <= 0.0)) || ((vel[1] <= 0.0) && (deltaX >= 0.0))
        yWindow[i] = ((vel[2] >= 0.0) && (deltaY <= 0.0)) || ((vel[2] <= 0.0) && (deltaY >= 0.0))
    end

    # X-derivative
    wVec_x = upwind.weightFunction(dxVec[xWindow], dyVec[xWindow]; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec[xWindow], dyVec[xWindow], wVec_x, dfVec[xWindow], upwind.res; order=upwind.order)
    ddx = upwind.res[1] / settings.interpRange
    if setCurvature && upwind.order == 2
        particleGrid.curvatures[particleIndex, 1] = upwind.res[3] / (settings.interpRange^2)
    end
        
    # Y-derivative
    wVec_y = upwind.weightFunction(dxVec[yWindow], dyVec[yWindow]; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec[yWindow], dyVec[yWindow], wVec_y, dfVec[yWindow], upwind.res; order=upwind.order)
    ddy = upwind.res[2] / settings.interpRange
    if setCurvature && upwind.order == 2
        particleGrid.curvatures[particleIndex, 2] = upwind.res[4] / (settings.interpRange^2)
    end

    return ddx * vel[1] + ddy * vel[2]
end


# ... (existing content of Interpolations.jl, including the UpwindWorkspace) ...

#==============================================================================
  Additional 2D Upwind Functors (Optimized for SoA Grids & Workspace)
==============================================================================#

function (upwind::UpwindGradient{PraveenAlgorithm})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector{<:Real}, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    vel = velocity(eq, 0.0)
    neighbour_indices = particleGrid.neighbour_indices[particleIndex]

    if setCurvature
        # Access curvature array directly. Assumes 2D curvature is stored as [dxx, dyy].
        particleGrid.curvatures[particleIndex, :] .= 0.0
    end
    
    # --- Least-Squares System Setup ---
    A11 = A12 = A22 = 0.0
    for nbIndex in neighbour_indices
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)
        A11 += w * (deltaX^2) 
        A12 += w * deltaX * deltaY
        A22 += w * (deltaY^2)
    end
    D = A11 * A22 - (A12^2)
    
    if abs(D) < 1e-14
        return 0.0 # Avoid division by zero if stencil is degenerate
    end
    
    # --- Divergence Calculation ---
    div = 0.0
    for nbIndex in neighbour_indices
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)
        
        # Solve for coefficients
        coeff_x = (A22 * w * deltaX - A12 * w * deltaY) / D
        coeff_y = (A11 * w * deltaY - A12 * w * deltaX) / D

        # Compute adapted coefficients for positivity
        angle = atan(deltaY, deltaX)
        n = (cos(angle), sin(angle))
        s = (-sin(angle), cos(angle))
        
        alfaBar = dot(n, (coeff_x, coeff_y))
        betaBar = dot(s, (coeff_x, coeff_y))
        
        bracketMinus = dot(vel, n) > 0.0 ? 0.0 : dot(vel, n)
        bracketMinus2 = betaBar * dot(vel, s) > 0.0 ? 0.0 : betaBar * dot(vel, s)
        
        cij = alfaBar * bracketMinus + bracketMinus2
        div += 2 * cij * (fVec[nbIndex] - fVec[particleIndex])
    end
    return div
end


function (upwind::UpwindGradient{NonLinearPraveenAlgorithm})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::Vector{<:Real}, 
    eq::ScalarHyperbolicPDE{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    neighbour_indices = particleGrid.neighbour_indices[particleIndex]

    if setCurvature
        particleGrid.curvatures[particleIndex, :] .= 0.0
    end
    
    # --- Least-Squares System Setup ---
    A11 = A12 = A22 = 0.0
    for nbIndex in neighbour_indices
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)
        A11 += w * (deltaX^2) 
        A12 += w * deltaX * deltaY
        A22 += w * (deltaY^2)
    end
    D = A11 * A22 - (A12^2)

    if abs(D) < 1e-14
        return 0.0
    end

    # --- Divergence Calculation ---
    ui = fVec[particleIndex]
    fiVec = flux(eq, ui)
    
    div = 0.0
    for nbIndex in neighbour_indices
        uj = fVec[nbIndex]
        deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)

        # Solve for coefficients
        coeff_x = (A22 * w * deltaX - A12 * w * deltaY) / D
        coeff_y = (A11 * w * deltaY - A12 * w * deltaX) / D
        
        # Compute adapted coefficients
        angle = atan(deltaY, deltaX)
        n = (cos(angle), sin(angle))
        s = (-sin(angle), cos(angle))

        Fni = dot(fiVec, n)
        Gsi = dot(fiVec, s)

        fjVec = flux(eq, uj)
        # Roe average speed, avoiding division by zero
        aij = (uj - ui) ≈ 0.0 ? velocity(eq, ui) : (fjVec .- fiVec) ./ (uj - ui)

        favg = (fiVec .+ fjVec) ./ 2.0
        
        # Rusanov flux for normal and tangential components
        Fnij = dot(favg, n) - 0.5 * abs(dot(aij, n)) * (uj - ui)
        Gsij = dot(favg, s) - 0.5 * sign(dot(s, (coeff_x, coeff_y))) * abs(dot(aij, s)) * (uj - ui)

        Fndiff = Fnij - Fni
        Gsdiff = Gsij - Gsi

        # Rotate flux differences back to Cartesian coordinates
        Fdiff = n[1] * Fndiff + s[1] * Gsdiff
        Gdiff = n[2] * Fndiff + s[2] * Gsdiff

        div += coeff_x * Fdiff + coeff_y * Gdiff
    end
    return 2 * div
end

