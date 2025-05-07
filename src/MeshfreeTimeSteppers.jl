export EulerUpwind, Upwind, RalstonRK2, RK3, RK4, MOODCriterion, MOODu1, MOODu2, NoMOOD, MOODLoubertU2, OnlyMOOD, FirstStageNoMOOD, MOODAlt

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
    deltaRelax::Bool
    function MOODu1(;deltaRelax::Bool)
        new(0, deltaRelax)
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
function (mood::MOODu1)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::Vector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    
    # Prep
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)
    d = maximum((particle.volume for particle in particleGrid.grid))
    δ = mood.deltaRelax ? d : 0.0;

    # MOOD
    moodEvent = (newRho < minU) || (newRho > maxU)
    moodEvent = moodEvent && (abs(maxU - minU) < δ^3) ? false : moodEvent  # If DMP Fail but flat section detection, don't do mood, else trust the standard DMP criterium.

    if firstStage
        particleGrid.grid[particleIndex].moodEvent = moodEvent
        if moodEvent && particleGrid.grid[particleIndex].moodEvent  # Check if there was at least one mood event in the previous stage 
            mood.count += 1
        end
    else
        particleGrid.grid[particleIndex].moodEvent = particleGrid.grid[particleIndex].moodEvent || moodEvent
    end
    return moodEvent
end

mutable struct MOODAlt <: MOODCriterion
    count::Int64
    deltaRelax::Bool
    function MOODAlt(;deltaRelax::Bool)
        new(0, deltaRelax)
    end
end

function (mood::MOODAlt)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::Vector{<:Real}, newRho::Real; firstStage = false)
    # Prep
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)
    d = maximum((particle.volume for particle in particleGrid.grid))
    δ = mood.deltaRelax ? d : 0.0;

    # MOOD
    moodEvent = (newRho < minU) || (newRho > maxU)
    moodEvent = moodEvent && (abs(maxU - minU) < δ^3) ? false : moodEvent
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
    deltaRelax::Bool
    function MOODLoubertU2(;deltaRelax::Bool)
        new(0, deltaRelax)
    end
end

function (mood::MOODLoubertU2)(particleGrid::ParticleGrid1D, particleIndex::Integer, rhoVec::Vector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    # Prep
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)
    d = maximum((particle.volume for particle in particleGrid.grid))
    δ = mood.deltaRelax ? d : 0.0;

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
    deltaRelax::Bool
    init::Bool
    delta::Float64
    function MOODu2(;deltaRelax::Bool)
        new(0, deltaRelax, false, 0.0)
    end
end

function (mood::MOODu2)(particleGrid::ParticleGridType, particleIndex::Integer, rhoVec::Vector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool where {ParticleGridType <: ParticleGrid}
    # Prep
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)

    if !mood.init
        d = maximum((particle.volume for particle in particleGrid.grid))
        mood.delta = mood.deltaRelax ? d : 0.0
        mood.init = true
    end

    # DMP criterion
    DMPFail = (newRho < minU) || (newRho > maxU)
    DMPFail = DMPFail && (abs(maxU - minU) < mood.delta^3) ? false : DMPFail
    
    # u2 check
    if ParticleGridType == ParticleGrid1D
        mini, maxi, minxx, maxxx = findLocalExtremaAbs!(particleGrid, particleIndex, particleGrid.temp)  # particleGrid.temp contains the curvatures
        u2 = (mini*maxi > -mood.delta) && ((minxx/maxxx >= 1.0 - (minxx/maxxx)^(1/1)) || (maxxx < mood.delta)) # True if criterion is satisfied, so no MOOD event
    elseif ParticleGridType == ParticleGrid2D
        mini1, maxi1, minxx1, maxxx1, mini2, maxi2, minxx2, maxxx2 = findLocalExtremaAbs!(particleGrid, particleIndex, particleGrid.temp)  # particleGrid.temp contains the curvatures
        u2x = (mini1*maxi1 > -mood.delta) && ((minxx1/maxxx1 >= 1/2) || (maxxx1 < mood.delta))
        u2y = (mini2*maxi2 > -mood.delta) && ((minxx2/maxxx2 >= 1/2) || (maxxx2 < mood.delta))
        u2 = u2x && u2y
    end
    
    # If DMP criterion failed, check u2 criterion
    moodEvent = DMPFail ? !u2 : false

    # Logging of MOOD events
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
    copyCurvatures!(particleGrid::ParticleGridType) where {ParticleGridType <: particleGrid}

Copies the curvatures of all particles in the grid to particleGrid.temp. This is then used by the u2 MOOD criterion.

"""
function copyCurvatures!(particleGrid::ParticleGridType) where {ParticleGridType <: ParticleGrid}
    if ParticleGridType == ParticleGrid1D
        map!(particle -> particle.curvature, particleGrid.temp, particleGrid.grid)
    elseif ParticleGridType == ParticleGrid2D
        for (i, particle) in enumerate(particleGrid.grid)
            particleGrid.temp[i, 1] = particle.curvature[1]
            particleGrid.temp[i, 2] = particle.curvature[2]
        end
    end
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

function (mood::NoMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::Vector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    particleGrid.grid[particleIndex].moodEvent = false
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
function (mood::OnlyMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::Vector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    particleGrid.grid[particleIndex].moodEvent = true
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
function (mood::FirstStageNoMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::Vector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    particleGrid.grid[particleIndex].moodEvent = !firstStage
    return !firstStage
end


# ------------------------------------------- Time steppers -------------------------------------------
struct EulerUpwind <: MeshfreeTimeStepper
    gradientInterpolator::GradientInterpolator
    rhoOld::Vector{Float64}
    function EulerUpwind(Nx::Integer; weightFunction = exponentialWeightFunction(), gradientInterpolator::GradientInterpolator = UpwindGradient(1; weightFunction))
        new(gradientInterpolator, Vector{Float64}(undef, Nx))
    end
    function EulerUpwind(Nx::Integer, Ny::Integer; algType::String = "Classic", weightFunction = exponentialWeightFunction()) 
        new(UpwindGradient(1; algType=algType, weightFunction=weightFunction), Vector{Float64}(undef, Nx*Ny))
    end
end

function (euler::EulerUpwind)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    map!(particle -> particle.rho, euler.rhoOld, particleGrid.grid)
    initTimeStep(euler.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        div = euler.gradientInterpolator(particleGrid, particleIndex, euler.rhoOld, eq, settings)
        particle.rho = particle.rho - div*dt
    end
end

function initTimeStepper(euler::EulerUpwind, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(euler.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
end

# struct EulerUpwindMood{G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
#     gradientInterpolator::G1
#     fallbackInterpolator::G2
#     mood::MOOD
#     rhoOld::Vector{Float64}
#     function EulerUpwind(Nx::Integer, gradientInterpolator::GradientInterpolator; fallbackInterpolator::GradientInterpolator = UpwindGradient(1), mood::MOODCriterion = NoMOOD())
#         new(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx))
#     end
#     function EulerUpwind(Nx::Integer, Ny::Integer; algType::String = "Classic", weightFunction = exponentialWeightFunction()) 
#         error("2D version not implemented with mood yet!")
#         new(UpwindGradient(1; algType=algType, weightFunction=weightFunction), Vector{Float64}(undef, Nx*Ny))
#     end
# end

# function (euler::EulerUpwindMood)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
#     map!(particle -> particle.rho, euler.rhoOld, particleGrid.grid)
#     initTimeStep(euler.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
#     for (particleIndex, particle) in enumerate(particleGrid.grid)
#         div = euler.gradientInterpolator(particleGrid, particleIndex, euler.rhoOld, eq, settings)
#         particle.rho = particle.rho - div*dt

#     end
# end

# function initTimeStepper(euler::EulerUpwindMood, particleGrid::ParticleGrid, settings::SimSetting)
#     initTimeStep(euler.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
# end

struct RK3{G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}
    div2::Vector{Float64}
    function RK3(gradientInterpolator::GradientInterpolator, Nx::Integer; fallbackInterpolator::GradientInterpolator = UpwindGradient(1), mood::MOODCriterion = NoMOOD())
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx))
    end
    function RK3(gradientInterpolator::GradientInterpolator, Nx::Integer, Ny::Integer; fallbackInterpolator::GradientInterpolator = UpwindGradient(1; algType="Praveen"), mood::MOODCriterion = NoMOOD())
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny))
    end
end

function initTimeStepper(rk3::RK3, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    initTimeStep(rk3.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)  # In case the fallbackInterpolator also starts populating the particle.alfaij fields, unpredictable things will start to happen.
end

function (rk3::RK3)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    # Fill stage 1
    map!(particle -> particle.rho, rk3.rhoInit, particleGrid.grid)
    copyCurvatures!(particleGrid)

    # Stage 2
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        rk3.div1[particleIndex] = rk3.gradientInterpolator(particleGrid, particleIndex, rk3.rhoInit, eq, settings)
        particle.rho = rk3.rhoInit[particleIndex] - rk3.div1[particleIndex]*dt/2
        if rk3.mood(particleGrid, particleIndex, rk3.rhoInit, particle.rho; firstStage=true)
            rk3.div1[particleIndex] = rk3.fallbackInterpolator(particleGrid, particleIndex, rk3.rhoInit, eq, settings; setCurvature=false)
            particle.rho = rk3.rhoInit[particleIndex] - rk3.div1[particleIndex]*dt/2  
        end
    end

    # Stage 3
    map!(particle -> particle.rho, rk3.rhos, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        rk3.div2[particleIndex] = rk3.gradientInterpolator(particleGrid, particleIndex, rk3.rhos, eq, settings)
        particle.rho = rk3.rhoInit[particleIndex] - dt*(2*rk3.div2[particleIndex] - rk3.div1[particleIndex])
        if rk3.mood(particleGrid, particleIndex, rk3.rhos, particle.rho)
            rk3.div2[particleIndex] = rk3.fallbackInterpolator(particleGrid, particleIndex, rk3.rhos, eq, settings; setCurvature=false)
            particle.rho = rk3.rhos[particleIndex] - dt*rk3.div2[particleIndex]  # FE step from previous stage
        end
    end

    # Final solution
    map!(particle -> particle.rho, rk3.rhos, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        grad = rk3.gradientInterpolator(particleGrid, particleIndex, rk3.rhos, eq, settings)
        particle.rho = rk3.rhoInit[particleIndex] - dt*(rk3.div1[particleIndex]/6 + 2*rk3.div2[particleIndex]/3 + grad/6)
        if rk3.mood(particleGrid, particleIndex, rk3.rhos, particle.rho)
            particle.rho = rk3.rhos[particleIndex]  # FE step from previous stage, but the previous stage is also the solution at final time (c_3 = 1.0). 
        end
    end
end

struct RalstonRK2{G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}
    function RalstonRK2(gradientInterpolator::GradientInterpolator, Nx::Integer; fallbackInterpolator::GradientInterpolator = UpwindGradient(1), mood::MOODCriterion = NoMOOD())
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx))
    end
    function RalstonRK2(gradientInterpolator::GradientInterpolator, Nx::Integer, Ny::Integer; fallbackInterpolator::GradientInterpolator = UpwindGradient(1; algType="Praveen"), mood::MOODCriterion = NoMOOD()) 
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny))
    end
end

function initTimeStepper(ralston::RalstonRK2, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)  # In case the fallbackInterpolator also starts populating the particle.alfaij fields, unpredictable things will start to happen.
end

# function (ralston::RalstonRK2)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
#     # First stage
#     map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid)
#     copyCurvatures!(particleGrid)
#     for (particleIndex, particle) in enumerate(particleGrid.grid)
#         ralston.div1[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings)
#         particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3
#         if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, particle.rho; firstStage=true)
#             ralston.div1[particleIndex] = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=false)
#             particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3
#         end
#     end

#     # Final stage
#     map!(particle -> particle.rho, ralston.rhos, particleGrid.grid)
#     copyCurvatures!(particleGrid)
#     for (particleIndex, particle) in enumerate(particleGrid.grid)
#         div = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings)
#         particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div/4)
#         if ralston.mood(particleGrid, particleIndex, ralston.rhos, particle.rho)
#             div = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings; setCurvature=false)
#             particle.rho = ralston.rhos[particleIndex] - dt*div/3
#             #particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div/4)
#         end
#     end
# end

# Alternative MOOD implementation with full RK1 fallback
function (ralston::RalstonRK2)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    # First stage
    map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        ralston.div1[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings)
        particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3
    end
    map!(particle -> particle.rho, ralston.rhos, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        div = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings)
        particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div/4)
    end

    for (particleIndex, particle) in enumerate(particleGrid.grid)
        if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, particle.rho; firstStage = true)
            div = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=false)
            particle.rho = ralston.rhoInit[particleIndex] - dt*div
        end
    end
end

# Gemini's function:
function (ralston::RalstonRK2)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = length(particleGrid.grid)
    k1_high = Vector{Float64}(undef, N)
    k2_high = Vector{Float64}(undef, N)
    k2_actual = Vector{Float64}(undef, N) # To store the chosen k2 tendency

    # Store initial solution u_n
    map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid)

    # --- Stage 1: Calculate k1 ---
    # Calculate curvatures based on u_n (rhoInit) if MOOD criterion needs them
    # The gradient interpolator itself should set particle.curvature
    # Then copyCurvatures! makes them available in particleGrid.temp for MOODu2
    # This assumes gradientInterpolator computes and sets curvatures.
    # If not, ensure curvatures are set based on ralston.rhoInit before the loop.
    # For safety, let's explicitly ensure curvatures are fresh for the first MOOD check.
    # This might involve calling the high-order gradient just to set curvatures
    # or having a separate curvature computation.
    # Assuming gradientInterpolator sets curvatures when called:
    for particleIndex in 1:N
        # Dummy call to set curvatures if not done by initTimeStepper or otherwise
        # This is a bit inefficient if gradientInterpolator is heavy.
        # A better approach would be a dedicated setCurvatures!(ralston.gradientInterpolator, ...)
        # or ensure initTimeStep also computes initial curvatures.
        # For now, assume curvatures are either up-to-date or will be set by the interpolator call.
        _ = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=true)
    end
    copyCurvatures!(particleGrid) # Make curvatures from particle.curvature available in particleGrid.temp for MOOD

    for particleIndex in 1:N
        k1_high[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=true) # ensure curvature is set by main interpolator

        # Candidate solution for u* (intermediate stage)
        # u_star_candidate = u_n - (2/3)*dt*k1_high
        rho_candidate_stage1 = ralston.rhoInit[particleIndex] - (2.0/3.0)*dt*k1_high[particleIndex]

        if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, rho_candidate_stage1; firstStage=true)
            ralston.div1[particleIndex] = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=false) # k1_actual
        else
            ralston.div1[particleIndex] = k1_high[particleIndex] # k1_actual
        end
    end

    # Compute u* using k1_actual (ralston.div1)
    # u* = u_n - (2/3)*dt*k1_actual
    for particleIndex in 1:N
        ralston.rhos[particleIndex] = ralston.rhoInit[particleIndex] - (2.0/3.0)*dt*ralston.div1[particleIndex]
    end

    # --- Stage 2: Calculate k2 ---
    # Calculate curvatures based on u* (ralston.rhos)
    for particleIndex in 1:N # Similar to above, ensure curvatures are set based on ralston.rhos
        _ = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings; setCurvature=true)
    end
    copyCurvatures!(particleGrid) # Make new curvatures available for MOOD

    for particleIndex in 1:N
        k2_high[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings; setCurvature=true) # ensure curvature is set

        # Candidate for final solution u_n+1
        # u_n+1_candidate = u_n - dt * ( (1/4)*k1_actual + (3/4)*k2_high )
        rho_candidate_final = ralston.rhoInit[particleIndex] - dt * ( (1.0/4.0)*ralston.div1[particleIndex] + (3.0/4.0)*k2_high[particleIndex] )

        # MOOD check using u_n (ralston.rhoInit) as the reference for DMP,
        # or ralston.rhos (u*) if that's more appropriate for your MOOD criterion's design.
        # The original code checked final rho against rhoInit.
        if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, rho_candidate_final; firstStage=false)
        # Alternative: if ralston.mood(particleGrid, particleIndex, ralston.rhos, rho_candidate_final; firstStage=false)
            k2_actual[particleIndex] = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings; setCurvature=false)
        else
            k2_actual[particleIndex] = k2_high[particleIndex]
        end
    end

    # --- Final Update ---
    # u_n+1 = u_n - dt * ( (1/4)*k1_actual + (3/4)*k2_actual )
    for particleIndex in 1:N
        particleGrid.grid[particleIndex].rho = ralston.rhoInit[particleIndex] - dt * ( (1.0/4.0)*ralston.div1[particleIndex] + (3.0/4.0)*k2_actual[particleIndex] )
    end
end

struct RK4{G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}
    div2::Vector{Float64}
    div3::Vector{Float64}
    function RK4(gradientInterpolator::GradientInterpolator, Nx::Integer; fallbackInterpolator::GradientInterpolator = UpwindGradient(1), mood::MOODCriterion = NoMOOD())
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx))
    end
    function RK4(gradientInterpolator::GradientInterpolator, Nx::Integer, Ny::Integer; fallbackInterpolator::GradientInterpolator = UpwindGradient(1; algType="Praveen"), mood::MOODCriterion = NoMOOD())
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny))
    end
end

function initTimeStepper(rk4::RK4, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)  # In case the fallbackInterpolator also starts populating the particle.alfaij fields, unpredictable things will start to happen.
end

function (rk4::RK4)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    # Fill stage 1
    map!(particle -> particle.rho, rk4.rhoInit, particleGrid.grid)    
    copyCurvatures!(particleGrid)

    # Stage 2
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        rk4.div1[particleIndex] = rk4.gradientInterpolator(particleGrid, particleIndex, rk4.rhoInit, eq, settings)
        particle.rho = rk4.rhoInit[particleIndex] - rk4.div1[particleIndex]*dt/2
        if rk4.mood(particleGrid, particleIndex, rk4.rhoInit, particle.rho; firstStage=true)
            rk4.div1[particleIndex] = rk4.fallbackInterpolator(particleGrid, particleIndex, rk4.rhoInit, eq, settings; setCurvature=false)
            particle.rho = rk4.rhoInit[particleIndex] - rk4.div1[particleIndex]*dt/2  
        end
    end

    # Stage 3
    map!(particle -> particle.rho, rk4.rhos, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        rk4.div2[particleIndex] = rk4.gradientInterpolator(particleGrid, particleIndex, rk4.rhos, eq, settings)
        particle.rho = rk4.rhoInit[particleIndex] - dt*rk4.div2[particleIndex]/2
        if rk4.mood(particleGrid, particleIndex, rk4.rhos, particle.rho)
            rk4.div2[particleIndex] = rk4.fallbackInterpolator(particleGrid, particleIndex, rk4.rhos, eq, settings; setCurvature=false)
            particle.rho = rk4.rhoInit[particleIndex] - dt*rk4.div2[particleIndex]/2  
        end
    end

    # Stage 3
    map!(particle -> particle.rho, rk4.rhos, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        rk4.div3[particleIndex] = rk4.gradientInterpolator(particleGrid, particleIndex, rk4.rhos, eq, settings)
        particle.rho = rk4.rhoInit[particleIndex] - dt*rk4.div3[particleIndex]
        if rk4.mood(particleGrid, particleIndex, rk4.rhos, particle.rho)
            rk4.div3[particleIndex] = rk4.fallbackInterpolator(particleGrid, particleIndex, rk4.rhos, eq, settings; setCurvature=false)
            particle.rho = rk4.rhoInit[particleIndex] - dt*rk4.div3[particleIndex]
        end
    end

    # Final solution
    map!(particle -> particle.rho, rk4.rhos, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        grad = rk4.gradientInterpolator(particleGrid, particleIndex, rk4.rhos, eq, settings)
        particle.rho = rk4.rhoInit[particleIndex] - dt*(rk4.div1[particleIndex]/6 + rk4.div2[particleIndex]/3 + rk4.div3[particleIndex]/3 + grad/6)
        if rk4.mood(particleGrid, particleIndex, rk4.rhos, particle.rho)
            particle.rho = rk4.rhos[particleIndex]  # Final stage is a solution at t^n+1 that satisfies the DMP
        end
    end
end
