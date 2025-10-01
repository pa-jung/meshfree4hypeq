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
    # For PraveenAlgorithm
    coeff_x_Vec::Vector{Float64}
    coeff_y_Vec::Vector{Float64}
    aij_x_Vec::Vector{Float64}
    aij_y_Vec::Vector{Float64}
    nxVec::Vector{Float64}
    nyVec::Vector{Float64}
    # Add a buffer for neighbor values
    ujVec::Vector{Float64}

    function UpwindWorkspace(max_neighbors::Int=30) # Preallocate with a reasonable capacity
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            falses(max_neighbors),
            falses(max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
        )
    end
end

# Helper to ensure workspace vectors are large enough
function ensure_capacity!(ws::UpwindWorkspace, n::Int)
    if length(ws.dxVec) < n
        resize!.((ws.dxVec, ws.dyVec, ws.dfVec, ws.wVec, ws.xWindow, ws.yWindow, ws.coeff_x_Vec, ws.coeff_y_Vec, ws.aij_x_Vec, ws.aij_y_Vec, ws.nxVec, ws.nyVec, ws.ujVec), n)
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
    
    upwind.weightFunction(wVec, dxVec; param=settings.interpAlpha, normalisation=1.0)
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

    weightFunction(wVec, dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
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
    upwind.weightFunction(wVec_x, dxVec[xWindow], dyVec[xWindow]; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec[xWindow], dyVec[xWindow], wVec_x, dfVec[xWindow], upwind.res; order=upwind.order)
    ddx = upwind.res[1] / settings.interpRange
    if setCurvature && upwind.order == 2
        particleGrid.curvatures[particleIndex, 1] = upwind.res[3] / (settings.interpRange^2)
    end
        
    # Y-derivative
    weightFunction(wVec_y, dxVec[yWindow], dyVec[yWindow]; param=settings.interpAlpha, normalisation=1.0)
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
    num_neighbors = length(neighbour_indices)
    
    if num_neighbors == 0; return 0.0; end

    if setCurvature
        particleGrid.curvatures[particleIndex, :] .= 0.0
    end
    
    # --- 1. Data Collection (Single Pass) ---
    ws = upwind.workspace
    ensure_capacity!(ws, num_neighbors)
    ui = fVec[particleIndex]
    for (i, nbIndex) in enumerate(neighbour_indices)
        ws.dxVec[i], ws.dyVec[i] = getDistance(particleGrid, particleIndex, nbIndex)
        ws.dfVec[i] = fVec[nbIndex] - ui
    end

    # Create views for the exact number of neighbors
    dxVec = @view ws.dxVec[1:num_neighbors]
    dyVec = @view ws.dyVec[1:num_neighbors]
    dfVec = @view ws.dfVec[1:num_neighbors]
    wVec  = @view ws.wVec[1:num_neighbors]
    
    # --- 2. Least-Squares System Setup (Vectorized) ---
    upwind.weightFunction(wVec, dxVec, dyVec; param=settings.interpAlpha, normalisation=settings.interpRange)

    A11 = sum(wVec[i] * dxVec[i]^2 for i in 1:num_neighbors)
    A22 = sum(wVec[i] * dyVec[i]^2 for i in 1:num_neighbors)
    A12 = sum(wVec[i] * dxVec[i] * dyVec[i] for i in 1:num_neighbors)
    D = A11 * A22 - A12^2
    
    if abs(D) < 1e-14; return 0.0; end
    
    # --- 3. Divergence Calculation (Vectorized) ---
    # Get views for all necessary buffers
    coeff_x_Vec = @view ws.coeff_x_Vec[1:num_neighbors]
    coeff_y_Vec = @view ws.coeff_y_Vec[1:num_neighbors]
    nxVec       = @view ws.nxVec[1:num_neighbors]
    nyVec       = @view ws.nyVec[1:num_neighbors]
    
    # Solve for coefficients (vectorized)
    coeff_x_Vec .= (wVec .* (A22 .* dxVec .- A12 .* dyVec)) ./ D
    coeff_y_Vec .= (wVec .* (A11 .* dyVec .- A12 .* dxVec)) ./ D

    # Rotational vectors (vectorized)
    angles = atan.(dyVec, dxVec)
    nxVec .= cos.(angles)
    nyVec .= sin.(angles)
    # sxVec is -nyVec, syVec is nxVec

    # Compute adapted coefficients and positivity terms (vectorized)
    # Reuse aij buffers for alfaBar and betaBar
    alfaBarVec = @view ws.aij_x_Vec[1:num_neighbors]
    betaBarVec = @view ws.aij_y_Vec[1:num_neighbors]
    
    alfaBarVec .= nxVec .* coeff_x_Vec .+ nyVec .* coeff_y_Vec
    betaBarVec .= (-nyVec) .* coeff_x_Vec .+ nxVec .* coeff_y_Vec
    
    # Reuse coeff buffers for temporary dot products with velocity
    dot_vel_n = coeff_x_Vec # Rename for clarity
    dot_vel_s = coeff_y_Vec
    
    dot_vel_n .= vel[1] .* nxVec .+ vel[2] .* nyVec
    dot_vel_s .= vel[1] .* (-nyVec) .+ vel[2] .* nxVec

    # Calculate bracket terms using min/max for conditional logic
    bracketMinusVec  = min.(dot_vel_n, 0.0)
    bracketMinus2Vec = min.(betaBarVec .* dot_vel_s, 0.0)
    
    # Calculate final coefficient vector `cij`
    # Reuse alfaBarVec buffer to store the final `cij` values
    cijVec = alfaBarVec
    cijVec .*= bracketMinusVec
    cijVec .+= bracketMinus2Vec
    
    # Final divergence is a single dot product
    div = 2 * dot(cijVec, dfVec)
    
    return div
end


function (upwind::UpwindGradient{NonLinearPraveenAlgorithm})(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    fVec::AbstractVector{<:Real}, 
    eq::LinearAdvection{2}, 
    settings::SimSetting; 
    setCurvature::Bool=true
)::Real
    
    vel = velocity(eq, 0.0)
    neighbour_indices = particleGrid.neighbour_indices[particleIndex]
    num_neighbors = length(neighbour_indices)
    
    if num_neighbors == 0; return 0.0; end

    if setCurvature
        particleGrid.curvatures[particleIndex, :] .= 0.0
    end
    
    # --- 1. Data Collection ---
    ws = upwind.workspace
    ensure_capacity!(ws, num_neighbors)
    ui = fVec[particleIndex]
    for (i, nbIndex) in enumerate(neighbour_indices)
        ws.dxVec[i], ws.dyVec[i] = getDistance(particleGrid, particleIndex, nbIndex)
        ws.ujVec[i] = fVec[nbIndex] # Store uj directly
    end

    # Create views
    dxVec = @view ws.dxVec[1:num_neighbors]
    dyVec = @view ws.dyVec[1:num_neighbors]
    ujVec = @view ws.ujVec[1:num_neighbors]
    wVec  = @view ws.wVec[1:num_neighbors]
    
    # --- 2. Least-Squares System ---
    upwind.weightFunction(wVec, dxVec, dyVec; param=settings.interpAlpha, normalisation=settings.interpRange)

    A11 = sum(wVec[i] * dxVec[i]^2 for i in 1:num_neighbors)
    A22 = sum(wVec[i] * dyVec[i]^2 for i in 1:num_neighbors)
    A12 = sum(wVec[i] * dxVec[i] * dyVec[i] for i in 1:num_neighbors)
    D = A11 * A22 - A12^2
    
    if abs(D) < 1e-14; return 0.0; end
    
    # --- 3. Divergence Calculation ---
    # Get views for buffers
    coeff_x_Vec = @view ws.coeff_x_Vec[1:num_neighbors]
    coeff_y_Vec = @view ws.coeff_y_Vec[1:num_neighbors]
    nxVec       = @view ws.nxVec[1:num_neighbors]
    nyVec       = @view ws.nyVec[1:num_neighbors]

    # --- THE FIX: Calculate flux components separately ---
    # For LinearAdvection, flux(u) = velocity * u. We can vectorize this directly.
    # Reuse aij buffers for the neighbor fluxes.
    fj_x_Vec = @view ws.aij_x_Vec[1:num_neighbors]
    fj_y_Vec = @view ws.aij_y_Vec[1:num_neighbors]

    fj_x_Vec .= vel[1] .* ujVec
    fj_y_Vec .= vel[2] .* ujVec
    
    # --- The rest of the vectorized logic follows ---

    # We need (uj - ui), which is (ujVec .- ui). Store it in dfVec.
    dfVec = @view ws.dfVec[1:num_neighbors]
    dfVec .= ujVec .- ui

    # The rest of the function remains the same, as it was already correct.
    coeff_x_Vec .= (wVec .* (A22 .* dxVec .- A12 .* dyVec)) ./ D
    coeff_y_Vec .= (wVec .* (A11 .* dyVec .- A12 .* dxVec)) ./ D
    
    angles = atan.(dyVec, dxVec)
    nxVec .= cos.(angles)
    nyVec .= sin.(angles)
    
    alfaBarVec = @view ws.aij_x_Vec[1:num_neighbors] # Reuse buffer
    betaBarVec = @view ws.aij_y_Vec[1:num_neighbors] # Reuse buffer
    
    alfaBarVec .= nxVec .* coeff_x_Vec .+ nyVec .* coeff_y_Vec
    betaBarVec .= (-nyVec) .* coeff_x_Vec .+ nxVec .* coeff_y_Vec
    
    dot_vel_n = coeff_x_Vec # Reuse buffer
    dot_vel_s = coeff_y_Vec # Reuse buffer
    
    dot_vel_n .= vel[1] .* nxVec .+ vel[2] .* nyVec
    dot_vel_s .= vel[1] .* (-nyVec) .+ vel[2] .* nxVec

    bracketMinusVec  = min.(dot_vel_n, 0.0)
    bracketMinus2Vec = min.(betaBarVec .* dot_vel_s, 0.0)
    
    cijVec = alfaBarVec # Reuse buffer
    cijVec .*= bracketMinusVec
    cijVec .+= bracketMinus2Vec
    
    div = 2 * dot(cijVec, dfVec)
    
    return div
end

