export EulerUpwind, Upwind, RalstonRK2, RK3, RK4, MOODCriterion, MOODu1, MOODu2, NoMOOD, MOODLoubertU2, OnlyMOOD, FirstStageNoMOOD, MOODAlt, RalstonRK2Limiter, RalstonRK2SmoothSwitch, RalstonRK2SmoothSwitch2

"""
    MOODCriterion

Abstract MOOD Criterion type. Each MOOD criterion should overload the ()-operator, checks if the MOOD criterion at that cell is satisfied.
Returns true for a MOOD event.
"""
abstract type MOODCriterion end

"""
    MOODu1

Original MOOD criterion. Checks DMP for rho. It has been observed that this limits the order to two.
particleGrid.grid[particleIndex].moodEvent is set to true if at least once during the RK-step a MOOD event occurs. 

# Arguments:
- `deltaRelax::Bool`: Relax DMP and other conditions in flat regions or not.
"""
mutable struct MOODu1 <: MOODCriterion 
    count::Int64
    deltaRelax::Float64
    function MOODu1(;deltaRelax::Real)
        new(0, convert(Float64, deltaRelax))
    end
end

"""
    (mood::MOODu1)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::Vector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool

# Arguments:
- `particleGrid::ParticleGrid`
- `particleIndex::Integer`
- `rhoVec::Vector{<:Real}`: Solutions at previous time step to be used in DMP.
- `newRho::Real`: newly proposed solution at next time step or RK stage.
- `firstStage::Bool`: True in case this is the first or only stage of the time integration routine. 
"""
# --- REFACTORED Functor for MOODu1 ---
# This method now works for any grid type thanks to the SoA design.
function (mood::MOODu1)(
    particleGrid::ParticleGrid, 
    particleIndex::Integer, 
    rhoVec::AbstractVector{<:Real}, 
    newRho::Real; 
    firstStage::Bool=false
)::Bool
    
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)
    
    # More efficient way to get max volume
    d = maximum(particleGrid.volumes) 
    δ = mood.deltaRelax * d

    moodEvent = (newRho < minU - δ) || (newRho > maxU + δ)
    
    # Check for flat sections (where DMP might fail spuriously)
    if abs(maxU - minU) < δ^3
        moodEvent = false
    end

    # Log the event to the grid's SoA boolean array
    if firstStage
        particleGrid.mood_events[particleIndex] = moodEvent
        if moodEvent
            mood.count += 1
        end
    else
        particleGrid.mood_events[particleIndex] = particleGrid.mood_events[particleIndex] || moodEvent
    end
    
    return moodEvent
end

mutable struct MOODAlt <: MOODCriterion
    count::Int64
    deltaRelax::Float64
    function MOODAlt(deltaRelax::Real)
        new(0, convert(Float64, deltaRelax))
    end
end

function (mood::MOODAlt)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage = false)
    # Prep
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)
    d = maximum((particle.volume for particle in particleGrid.grid))

    # MOOD
    moodEvent = (newRho < minU) || (newRho > maxU)
    moodEvent = moodEvent && (abs(maxU - minU) < mood.deltaRelax * d^3) ? false : moodEvent
    particleGrid.grid[particleIndex].moodEvent = moodEvent
    if moodEvent
        mood.count += 1
    end
    return moodEvent

end


"""
    MOODLoubertU2

Enhanced MOOD criterion. Checks relaxed DMP for rho. This criterion relaxes the DMP at local extrema using second order derivatives.
"""
mutable struct MOODLoubertU2 <: MOODCriterion
    count::Int64
    deltaRelax::Float64
    function MOODLoubertU2(;deltaRelax::Real)
        new(0, convert(Float64, deltaRelax))
    end
end

function (mood::MOODLoubertU2)(particleGrid::ParticleGrid1D, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    # Prep
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)
    d = maximum((particle.volume for particle in particleGrid.grid))
    δ = mood.deltaRelax * d#mood.deltaRelax ? d : 0.0;

    # DMP criterion
    DMPFail = (newRho < minU) || (newRho > maxU)
    DMPFail = DMPFail && (abs(maxU - minU) < δ^3) ? false : DMPFail
    
    # u2 check with faulty curvatures!
    mini, maxi = findLocalExtrema!(particleGrid, particleIndex, particleGrid.temp)  # particleGrid.temp contains the curvatures
    u2 = (mini*maxi > -δ) && ((mini/maxi >= 0.5) || (max(abs(mini), abs(maxi)) < δ)) # True if criterion is satisfied, so no MOOD event

    # If DMP criterion failed, check u2 criterion
    moodEvent = DMPFail ? !u2 : false
    if firstStage
        if particleGrid.grid[particleIndex].moodEvent  # Check if there was at least one mood event in the previous stage 
            mood.count += 1
        end
        particleGrid.grid[particleIndex].moodEvent = moodEvent
    else
        particleGrid.grid[particleIndex].moodEvent = particleGrid.grid[particleIndex].moodEvent || moodEvent
    end
    return moodEvent
end

"""
    MOODu2

Enhanced MOOD criterion. Checks relaxed DMP for rho. This criterion relaxes the DMP at local extrema using second order derivatives.
"""
mutable struct MOODu2 <: MOODCriterion 
    count::Int64
    deltaRelax::Float64
    init::Bool
    delta::Float64
    function MOODu2(;deltaRelax::Real)
        new(0, convert(Float64, deltaRelax), false, 0.0)
    end
end

# --- REFACTORED Functor for MOODu2 ---
# We use dispatch to create separate, clear methods for 1D and 2D.
function (mood::MOODu2)(
    particleGrid::ParticleGrid1D, 
    particleIndex::Integer, 
    rhoVec::AbstractVector{<:Real}, 
    newRho::Real; 
    firstStage::Bool=false
)::Bool
    
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)

    if !mood.init
        d = maximum(particleGrid.volumes)
        mood.delta = mood.deltaRelax * d
        mood.init = true
    end

    DMPFail = (newRho < minU - mood.delta) || (newRho > maxU + mood.delta)
    if abs(maxU - minU) < mood.delta^3
        DMPFail = false
    end
    
    # u2 check for 1D
    # particleGrid.temp now holds curvatures from `copyCurvatures!`
    mini, maxi, minxx, maxxx = findLocalExtremaAbs!(particleGrid, particleIndex, particleGrid.temp)
    u2_satisfied = (mini * maxi > -mood.delta) && ((minxx / maxxx >= 0.5) || (maxxx < mood.delta))
    
    moodEvent = DMPFail ? !u2_satisfied : false

    if firstStage
        particleGrid.mood_events[particleIndex] = moodEvent
        if moodEvent
            mood.count += 1
        end
    else
        particleGrid.mood_events[particleIndex] = particleGrid.mood_events[particleIndex] || moodEvent
    end
    
    return moodEvent
end

function (mood::MOODu2)(
    particleGrid::ParticleGrid2D, 
    particleIndex::Integer, 
    rhoVec::AbstractVector{<:Real}, 
    newRho::Real; 
    firstStage::Bool=false
)::Bool
    
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)

    if !mood.init
        d = maximum(particleGrid.volumes)
        mood.delta = mood.deltaRelax * d
        mood.init = true
    end

    DMPFail = (newRho < minU - mood.delta) || (newRho > maxU + mood.delta)
    if abs(maxU - minU) < mood.delta^3
        DMPFail = false
    end
    
    # u2 check for 2D
    # particleGrid.temp is now an N x 2 matrix of curvatures
    # We need a new findLocalExtremaAbs! that works on this SoA data
    extrema_vals = findLocalExtremaAbs!(particleGrid, particleIndex, particleGrid.temp)
    mini1, maxi1, minxx1, maxxx1 = extrema_vals[1:4]
    mini2, maxi2, minxx2, maxxx2 = extrema_vals[5:8]
    
    u2x = (mini1 * maxi1 > -mood.delta) && ((minxx1 / maxxx1 >= 0.5) || (maxxx1 < mood.delta))
    u2y = (mini2 * maxi2 > -mood.delta) && ((minxx2 / maxxx2 >= 0.5) || (maxxx2 < mood.delta))
    u2_satisfied = u2x && u2y
    
    moodEvent = DMPFail ? !u2_satisfied : false

    if firstStage
        particleGrid.mood_events[particleIndex] = moodEvent
        if moodEvent
            mood.count += 1
        end
    else
        particleGrid.mood_events[particleIndex] = particleGrid.mood_events[particleIndex] || moodEvent
    end
    
    return moodEvent
end

# --- REFACTORED copyCurvatures! ---
# We use dispatch for clean 1D/2D implementations.
function copyCurvatures!(particleGrid::ParticleGrid1D)
    # Direct array copy is much faster than `map!`
    particleGrid.temp .= particleGrid.curvatures
end

function copyCurvatures!(particleGrid::ParticleGrid2D)
    # Direct array copy for the 2D matrix of curvatures
    particleGrid.temp .= particleGrid.curvatures
end

"""
    NoMOOD

No MOOD. Results in a standard time integration routine.
"""
struct NoMOOD <: MOODCriterion 
    count::Int64
    function NoMOOD()
        new(0)
    end
end

"""
OnlyMOOD

Test case for always using the fallback Interpolator
"""
struct OnlyMOOD <: MOODCriterion
    count::Int64
    function OnlyMOOD()
        new(0)
    end
end

# --- REFACTORED NoMOOD/OnlyMOOD Functors ---
# These are updated to write to the new `mood_events` SoA array.
function (mood::NoMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    particleGrid.mood_events[particleIndex] = false
    return false
end

function (mood::OnlyMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    particleGrid.mood_events[particleIndex] = true
    return true
end

"""
FirstStageNoMOOD

Test case for mixing of stages. Returns false only in the first stage.
"""
struct FirstStageNoMOOD <: MOODCriterion
    count::Int64
    function FirstStageNoMOOD()
        new(0)
    end
end
function (mood::FirstStageNoMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    particleGrid.grid[particleIndex].moodEvent = !firstStage
    return !firstStage
end


# A simple example for EulerUpwind
struct EulerUpwind{G <: GradientInterpolator} <: MeshfreeTimeStepper
    gradientInterpolator::G
end

function (eu::EulerUpwind)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    rho_n = copy(particleGrid.rhos) # Store initial state for the step
    interior = particleGrid.interior_indices
    
    initTimeStep(eu.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)

    for p_idx in interior
        div = eu.gradientInterpolator(particleGrid, p_idx, rho_n, eq, settings)
        particleGrid.rhos[p_idx] = rho_n[p_idx] - dt * div
    end
end

function initTimeStepper(euler::EulerUpwind, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(euler.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
end

struct RK3{G1, G2, MOOD} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}      # Stores the solution at the start of the step
    rho_stage1::Vector{Float64} # Stores the result of the first stage
    rho_stage2::Vector{Float64} # Stores the result of the second stage
    
    div1::Vector{Float64} # Stores divergence from stage 1
    div2::Vector{Float64} # Stores divergence from stage 2
    div3::Vector{Float64} # Stores divergence from stage 3

    function RK3(grad::G1, fallback::G2, mood::M) where {G1, G2, M}
        # Initialize with empty buffers; they will be resized on the first call
        new{G1, G2, M}(grad, fallback, mood, Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

# --- User-Friendly Constructor ---
function RK3(gradientInterpolator::G1; fallbackInterpolator::G2 = nothing, mood::M = NoMOOD()) where {G1, G2, M}
    RK3(gradientInterpolator, fallbackInterpolator, mood)
end

function initTimeStepper(rk3::RK3, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk3.fallbackInterpolator)
        initTimeStep(rk3.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
end

function (rk3::RK3)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    # --- Ensure buffers are correctly sized for the current grid ---
    if length(rk3.rho_n) != N
        resize!.((rk3.rho_n, rk3.rho_stage1, rk3.rho_stage2, rk3.div1, rk3.div2, rk3.div3), N)
    end

    interior = particleGrid.interior_indices
    rk3.rho_n .= particleGrid.rhos # Store u^n

    # --- Stage 1: u^(1) = u^n + dt * L(u^n) ---
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk3.fallbackInterpolator); initTimeStep(rk3.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange); end

    for p_idx in interior
        # L(u^n) is -div(u^n)
        rk3.div1[p_idx] = rk3.gradientInterpolator(particleGrid, p_idx, rk3.rho_n, eq, settings)
        
        # Calculate candidate for stage 1
        rho_candidate = rk3.rho_n[p_idx] - dt * rk3.div1[p_idx]
        
        # Apply MOOD if necessary
        if !isnothing(rk3.fallbackInterpolator) && rk3.mood(particleGrid, p_idx, rk3.rho_n, rho_candidate; firstStage=true)
            rk3.div1[p_idx] = rk3.fallbackInterpolator(particleGrid, p_idx, rk3.rho_n, eq, settings; setCurvature=false)
            rho_candidate = rk3.rho_n[p_idx] - dt * rk3.div1[p_idx]
        end
        rk3.rho_stage1[p_idx] = rho_candidate
    end
    particleGrid.rhos[interior] .= @view rk3.rho_stage1[interior]
    apply_boundary_conditions!(particleGrid) # CRITICAL: Update ghost cells for next stage
    rk3.rho_stage1 .= particleGrid.rhos # Update buffer with correct ghost cells

    # --- Stage 2: u^(2) = 3/4 u^n + 1/4 u^(1) + 1/4 dt * L(u^(1)) ---
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    
    for p_idx in interior
        rk3.div2[p_idx] = rk3.gradientInterpolator(particleGrid, p_idx, rk3.rho_stage1, eq, settings)
        
        rho_candidate = 0.75 * rk3.rho_n[p_idx] + 0.25 * rk3.rho_stage1[p_idx] - 0.25 * dt * rk3.div2[p_idx]
        
        if !isnothing(rk3.fallbackInterpolator) && rk3.mood(particleGrid, p_idx, rk3.rho_stage1, rho_candidate)
            rk3.div2[p_idx] = rk3.fallbackInterpolator(particleGrid, p_idx, rk3.rho_stage1, eq, settings; setCurvature=false)
            rho_candidate = 0.75 * rk3.rho_n[p_idx] + 0.25 * rk3.rho_stage1[p_idx] - 0.25 * dt * rk3.div2[p_idx]
        end
        rk3.rho_stage2[p_idx] = rho_candidate
    end
    particleGrid.rhos[interior] .= @view rk3.rho_stage2[interior]
    apply_boundary_conditions!(particleGrid) # CRITICAL: Update ghost cells for next stage
    rk3.rho_stage2 .= particleGrid.rhos

    # --- Stage 3 (Final): u^{n+1} = 1/3 u^n + 2/3 u^(2) + 2/3 dt * L(u^(2)) ---
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)

    for p_idx in interior
        rk3.div3[p_idx] = rk3.gradientInterpolator(particleGrid, p_idx, rk3.rho_stage2, eq, settings)

        rho_final = (1/3) * rk3.rho_n[p_idx] + (2/3) * rk3.rho_stage2[p_idx] - (2/3) * dt * rk3.div3[p_idx]
        
        if !isnothing(rk3.fallbackInterpolator) && rk3.mood(particleGrid, p_idx, rk3.rho_stage2, rho_final)
            rk3.div3[p_idx] = rk3.fallbackInterpolator(particleGrid, p_idx, rk3.rho_stage2, eq, settings; setCurvature=false)
            rho_final = (1/3) * rk3.rho_n[p_idx] + (2/3) * rk3.rho_stage2[p_idx] - (2/3) * dt * rk3.div3[p_idx]
        end
        particleGrid.rhos[p_idx] = rho_final
    end
end

struct RK4{G1, G2, MOOD} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_stage::Vector{Float64} # A single buffer for all intermediate stages
    
    k1::Vector{Float64} # Stores divergence from stage 1
    k2::Vector{Float64} # Stores divergence from stage 2
    k3::Vector{Float64} # Stores divergence from stage 3
    k4::Vector{Float64} # Stores divergence from stage 4

    function RK4(grad::G1, fallback::G2, mood::M) where {G1, G2, M}
        # Initialize with empty buffers; they will be resized on the first call
        new{G1, G2, M}(grad, fallback, mood, Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

# --- User-Friendly Constructor ---
function RK4(gradientInterpolator::G1; fallbackInterpolator::G2 = nothing, mood::M = NoMOOD()) where {G1, G2, M}
    RK4(gradientInterpolator, fallbackInterpolator, mood)
end

function initTimeStepper(rk4::RK4, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk4.fallbackInterpolator)
        initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
end

function (rk4::RK4)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    # --- Ensure buffers are correctly sized for the current grid ---
    if length(rk4.rho_n) != N
        resize!.((rk4.rho_n, rk4.rho_stage, rk4.k1, rk4.k2, rk4.k3, rk4.k4), N)
    end

    interior = particleGrid.interior_indices
    rk4.rho_n .= particleGrid.rhos # Store u^n

    # --- Stage 1: Calculate k1 = -div(u^n) ---
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk4.fallbackInterpolator); initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange); end

    for p_idx in interior
        rk4.k1[p_idx] = rk4.gradientInterpolator(particleGrid, p_idx, rk4.rho_n, eq, settings)
        
        # MOOD check is for the candidate solution of the *next* stage
        rho_candidate = rk4.rho_n[p_idx] - 0.5 * dt * rk4.k1[p_idx]
        if !isnothing(rk4.fallbackInterpolator) && rk4.mood(particleGrid, p_idx, rk4.rho_n, rho_candidate; firstStage=true)
            rk4.k1[p_idx] = rk4.fallbackInterpolator(particleGrid, p_idx, rk4.rho_n, eq, settings; setCurvature=false)
        end
    end

    # --- Stage 2: Calculate k2 = -div(u^n + 0.5*dt*k1) ---
    @. rk4.rho_stage = rk4.rho_n - 0.5 * dt * rk4.k1
    particleGrid.rhos[interior] .= @view rk4.rho_stage[interior]
    apply_boundary_conditions!(particleGrid)
    rk4.rho_stage .= particleGrid.rhos # Update buffer with correct ghosts

    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for p_idx in interior
        rk4.k2[p_idx] = rk4.gradientInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings)
        rho_candidate = rk4.rho_n[p_idx] - 0.5 * dt * rk4.k2[p_idx]
        if !isnothing(rk4.fallbackInterpolator) && rk4.mood(particleGrid, p_idx, rk4.rho_stage, rho_candidate)
            rk4.k2[p_idx] = rk4.fallbackInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings; setCurvature=false)
        end
    end

    # --- Stage 3: Calculate k3 = -div(u^n + 0.5*dt*k2) ---
    @. rk4.rho_stage = rk4.rho_n - 0.5 * dt * rk4.k2
    particleGrid.rhos[interior] .= @view rk4.rho_stage[interior]
    apply_boundary_conditions!(particleGrid)
    rk4.rho_stage .= particleGrid.rhos

    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for p_idx in interior
        rk4.k3[p_idx] = rk4.gradientInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings)
        rho_candidate = rk4.rho_n[p_idx] - dt * rk4.k3[p_idx]
        if !isnothing(rk4.fallbackInterpolator) && rk4.mood(particleGrid, p_idx, rk4.rho_stage, rho_candidate)
            rk4.k3[p_idx] = rk4.fallbackInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings; setCurvature=false)
        end
    end

    # --- Stage 4: Calculate k4 = -div(u^n + dt*k3) ---
    @. rk4.rho_stage = rk4.rho_n - dt * rk4.k3
    particleGrid.rhos[interior] .= @view rk4.rho_stage[interior]
    apply_boundary_conditions!(particleGrid)
    rk4.rho_stage .= particleGrid.rhos

    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for p_idx in interior
        rk4.k4[p_idx] = rk4.gradientInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings)
        # Final MOOD check
        rho_final = rk4.rho_n[p_idx] - (dt/6) * (rk4.k1[p_idx] + 2*rk4.k2[p_idx] + 2*rk4.k3[p_idx] + rk4.k4[p_idx])
        if !isnothing(rk4.fallbackInterpolator) && rk4.mood(particleGrid, p_idx, rk4.rho_stage, rho_final)
            # If the final step fails, a common fallback is to take a first-order Euler step
            # using the final stage's divergence (k4). This is a robust choice.
            particleGrid.rhos[p_idx] = rk4.rho_stage[p_idx] - dt * rk4.k4[p_idx]
        else
            particleGrid.rhos[p_idx] = rho_final
        end
    end
end


# No longer needs Nx, Ny. Buffers are sized based on the grid passed during the call.
struct RalstonRK2{G1, G2, MOOD} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    # Buffers are now part of the struct to be reused
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}

    function RalstonRK2(grad::G1, fallback::G2, mood::M) where {G1, G2, M}
        # Initialize with empty buffers, they will be resized on the first step
        new{G1, G2, M}(grad, fallback, mood, Float64[], Float64[], Float64[])
    end
end

# User-friendly constructor
function RalstonRK2(gradientInterpolator::G1; fallbackInterpolator::G2 = nothing, mood::M = NoMOOD()) where {G1, G2, M}
    RalstonRK2(gradientInterpolator, fallbackInterpolator, mood)
end


function (ralston::RalstonRK2)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    # --- Resize buffers only if necessary ---
    if length(ralston.rhoInit) != N
        resize!.((ralston.rhoInit, ralston.rhos, ralston.div1), N)
    end

    interior = particleGrid.interior_indices
    
    # First stage
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(ralston.fallbackInterpolator)
        initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
    
    ralston.rhoInit .= particleGrid.rhos # Use the SoA rhos array
    
    for p_idx in interior
        ralston.div1[p_idx] = ralston.gradientInterpolator(particleGrid, p_idx, ralston.rhoInit, eq, settings)
        
        rho_candidate = ralston.rhoInit[p_idx] - ralston.div1[p_idx] * dt * 2/3
        
        if !isnothing(ralston.fallbackInterpolator) && ralston.mood(particleGrid, p_idx, ralston.rhoInit, rho_candidate; firstStage=true)
            ralston.div1[p_idx] = ralston.fallbackInterpolator(particleGrid, p_idx, ralston.rhoInit, eq, settings; setCurvature=false)
            rho_candidate = ralston.rhoInit[p_idx] - ralston.div1[p_idx] * dt * 2/3
        end
        ralston.rhos[p_idx] = rho_candidate # Store intermediate result in rhos buffer
    end
    # Update physical grid with intermediate stage (only for interior)
    particleGrid.rhos[interior] .= @view ralston.rhos[interior]

    # Final stage
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    
    for p_idx in interior
        div2 = ralston.gradientInterpolator(particleGrid, p_idx, ralston.rhos, eq, settings)
        
        rho_final = ralston.rhoInit[p_idx] - dt * (ralston.div1[p_idx] / 4 + 3 * div2 / 4)

        if !isnothing(ralston.fallbackInterpolator) && ralston.mood(particleGrid, p_idx, ralston.rhos, rho_final)
            div2 = ralston.fallbackInterpolator(particleGrid, p_idx, ralston.rhos, eq, settings; setCurvature=false)
            rho_final = ralston.rhoInit[p_idx] - dt * (ralston.div1[p_idx] / 4 + 3 * div2 / 4)
        end
        particleGrid.rhos[p_idx] = rho_final # Write final result to the grid
    end
end

struct RalstonRK2SmoothSwitch{G1, G2, MOOD} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    tol::Float64
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_stage::Vector{Float64}
    rho_fallback::Vector{Float64}
    div1::Vector{Float64}
    
    # --- Propagation Buffers ---
    mood_indices::Vector{Int}
    prop_indices::Vector{Int}
    
    # Per-step flag to track which particles have been switched to fallback
    switched_to_fallback::BitVector

    function RalstonRK2SmoothSwitch(grad::G1, fallback::G2, mood::M; tol=1e-7) where {G1, G2, M}
        # Initialize with empty buffers; they will be resized on the first call
        new{G1, G2, M}(grad, fallback, mood, tol,
            Float64[], Float64[], Float64[], Float64[], # Main buffers
            Int[], Int[], # Propagation buffers
            falses(0)    # Flag buffer
        )
    end
end

# --- User-Friendly Constructor ---
function RalstonRK2SmoothSwitch(gradientInterpolator::G1; fallbackInterpolator::G2 = gradientInterpolator, mood::M = NoMOOD(), tol = 1e-7) where {G1, G2, M}
    RalstonRK2SmoothSwitch(gradientInterpolator, fallbackInterpolator, mood; tol=tol)
end

function initTimeStepper(ralston::RalstonRK2SmoothSwitch, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
end

function (ralston::RalstonRK2SmoothSwitch)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    # --- Ensure buffers are correctly sized for the current grid ---
    if length(ralston.rho_n) != N
        resize!.((ralston.rho_n, ralston.rho_stage, ralston.rho_fallback, ralston.div1), N)
        resize!(ralston.switched_to_fallback, N)
    end

    interior = particleGrid.interior_indices
    ralston.rho_n .= particleGrid.rhos # Store u^n
    
    # --- 1. Calculate Full Fallback Solution and Target Mass ---
    initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    target_mass = 0.0
    for p_idx in interior
        div_fallback = ralston.fallbackInterpolator(particleGrid, p_idx, ralston.rho_n, eq, settings; setCurvature=false)
        ralston.rho_fallback[p_idx] = ralston.rho_n[p_idx] - div_fallback * dt
        target_mass += ralston.rho_fallback[p_idx] * particleGrid.volumes[p_idx]
    end

    # --- 2. Perform High-Order RalstonRK2 Step ---
    # First stage
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for p_idx in interior
        ralston.div1[p_idx] = ralston.gradientInterpolator(particleGrid, p_idx, ralston.rho_n, eq, settings)
        ralston.rho_stage[p_idx] = ralston.rho_n[p_idx] - ralston.div1[p_idx] * dt * 2/3
    end
    particleGrid.rhos[interior] .= @view ralston.rho_stage[interior]
    apply_boundary_conditions!(particleGrid)
    ralston.rho_stage .= particleGrid.rhos

    # Final stage
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    current_mass = 0.0
    empty!(ralston.mood_indices)
    fill!(ralston.switched_to_fallback, false)

    for p_idx in interior
        div2 = ralston.gradientInterpolator(particleGrid, p_idx, ralston.rho_stage, eq, settings)
        rho_final_candidate = ralston.rho_n[p_idx] - dt * (ralston.div1[p_idx]/4 + 3*div2/4)
        
        # --- 3. Initial MOOD Check ---
        if ralston.mood(particleGrid, p_idx, ralston.rho_n, rho_final_candidate; firstStage=true)
            particleGrid.rhos[p_idx] = ralston.rho_fallback[p_idx]
            push!(ralston.mood_indices, p_idx)
            ralston.switched_to_fallback[p_idx] = true
        else
            particleGrid.rhos[p_idx] = rho_final_candidate
        end
        current_mass += particleGrid.rhos[p_idx] * particleGrid.volumes[p_idx]
    end

    # --- 4. Mass Conservation Propagation Loop ---
    if !isempty(ralston.mood_indices)
        # Build the initial propagation list from neighbors of MOOD events
        empty!(ralston.prop_indices)
        for p_idx in ralston.mood_indices
            for nb_idx in particleGrid.neighbour_indices[p_idx]
                # Only add interior neighbors that haven't been switched yet
                if nb_idx in interior && !ralston.switched_to_fallback[nb_idx]
                    push!(ralston.prop_indices, nb_idx)
                end
            end
        end
        unique!(ralston.prop_indices) # Remove duplicates

        while abs(target_mass - current_mass) > ralston.tol && !isempty(ralston.prop_indices)
            p_idx = popfirst!(ralston.prop_indices)
            
            # This check is redundant if we filter when adding, but safe
            if ralston.switched_to_fallback[p_idx]; continue; end
            
            # Switch this particle to the low-order solution
            local_mass_change = (ralston.rho_fallback[p_idx] - particleGrid.rhos[p_idx]) * particleGrid.volumes[p_idx]
            current_mass += local_mass_change
            particleGrid.rhos[p_idx] = ralston.rho_fallback[p_idx]
            ralston.switched_to_fallback[p_idx] = true

            # Add its neighbors to the propagation list
            for nb_idx in particleGrid.neighbour_indices[p_idx]
                if nb_idx in interior && !ralston.switched_to_fallback[nb_idx]
                    push!(ralston.prop_indices, nb_idx)
                end
            end
        end
    end
end
