module MOOD

using ..ParticleGrids


export MOODCriterion, MOODu1, MOODu2, NoMOOD, MOODLoubertU2, OnlyMOOD, FirstStageNoMOOD, initMOOD!

"""
    MOODCriterion

Abstract MOOD Criterion type. Each MOOD criterion should overload the ()-operator, checks if the MOOD criterion at that cell is satisfied.
Returns true for a MOOD event.
"""
abstract type MOODCriterion end

function initMOOD!(mood::MOODCriterion, d)
    return
end

"""
    MOODu1

Original MOOD criterion. Checks DMP for rho. It has been observed that this limits the order to two.
particleGrid.grid[particleIndex].moodEvent is set to true if at least once during the RK-step a MOOD event occurs. 

# Arguments:
- `deltaRelax::Bool`: Relax DMP and other conditions in flat regions or not.
"""
mutable struct MOODu1 <: MOODCriterion 
    count::Int64
    const d::Float64
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
    particleGrid::ParticleGrid{D}, 
    particleIndex::Int, 
    rhoVec::AbstractVector{Float64}, 
    newRho::Float64; 
    firstStage::Bool=false
)::Bool where {D}
    
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)
    # More efficient way to get max volume
    δ = mood.d

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


"""
    MOODLoubertU2

Enhanced MOOD criterion. Checks relaxed DMP for rho. This criterion relaxes the DMP at local extrema using second order derivatives.
"""
mutable struct MOODLoubertU2 <: MOODCriterion
    count::Int64
    const d::Float64
    function MOODLoubertU2(;deltaRelax::Real)
        new(0, convert(Float64, deltaRelax))
    end
end

function (mood::MOODLoubertU2)(particleGrid::ParticleGrid1D, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    # Prep
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)
    δ = mood.d

    # DMP criterion
    DMPFail = (newRho < minU) || (newRho > maxU)
    DMPFail = DMPFail && (abs(maxU - minU) < δ^3) ? false : DMPFail
    
    # u2 check with faulty curvatures!
    mini, maxi = findLocalExtrema!(particleGrid, particleIndex, particleGrid.curvatures)  # particleGrid.curvatures contains the curvatures
    u2 = (mini*maxi > -δ) && ((mini/maxi >= 0.5) || (max(abs(mini), abs(maxi)) < δ)) # True if criterion is satisfied, so no MOOD event

    # If DMP criterion failed, check u2 criterion
    moodEvent = DMPFail ? !u2 : false
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

"""
    MOODu2

Enhanced MOOD criterion. Checks relaxed DMP for rho. This criterion relaxes the DMP at local extrema using second order derivatives.
"""
mutable struct MOODu2 <: MOODCriterion 
    count::Int64
    const d::Float64
    function MOODu2(;deltaRelax::Real)
        new(0, convert(Float64, deltaRelax))
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

    delta = mood.d

    DMPFail = (newRho < minU - delta) || (newRho > maxU + delta)
    if abs(maxU - minU) < delta^3
        DMPFail = false
    end
    
    # u2 check for 1D
    # particleGrid.curvatures now holds curvatures from `copyCurvatures!`
    # @code_warntype run_extrema_test(particleGrid, particleIndex, particleGrid.curvatures)
    # error("TEST!")
    mini, maxi, minxx, maxxx = findLocalExtremaAbs!(particleGrid, particleIndex, particleGrid.curvatures)
    u2_satisfied = (mini * maxi > -delta) && ((minxx / maxxx >= 0.5) || (maxxx < delta))
    
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
    delta = mood.d

    DMPFail = (newRho < minU - delta) || (newRho > maxU + delta)
    if abs(maxU - minU) < delta^3
        DMPFail = false
    end
    
    # u2 check for 2D
    # particleGrid.curvatures is now an N x 2 matrix of curvatures
    # We need a new findLocalExtremaAbs! that works on this SoA data
    extrema_vals = findLocalExtremaAbs!(particleGrid, particleIndex, particleGrid.curvatures)
    mini1, maxi1, minxx1, maxxx1 = extrema_vals[1:4]
    mini2, maxi2, minxx2, maxxx2 = extrema_vals[5:8]
    
    u2x = (mini1 * maxi1 > -delta) && ((minxx1 / maxxx1 >= 0.5) || (maxxx1 < delta))
    u2y = (mini2 * maxi2 > -delta) && ((minxx2 / maxxx2 >= 0.5) || (maxxx2 < delta))
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

function (mood::NoMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    particleGrid.mood_events[particleIndex] = false
    return false
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


end