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
function (mood::MOODu1)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
    
    # Prep
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)
    d = maximum((particle.volume for particle in particleGrid.grid))
    δ = mood.deltaRelax * d

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

function (mood::MOODu2)(particleGrid::ParticleGridType, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool where {ParticleGridType <: ParticleGrid}
    # Prep
    minU, maxU = findLocalExtrema!(particleGrid, particleIndex, rhoVec)

    if !mood.init
        d = maximum((particle.volume for particle in particleGrid.grid))
        mood.delta = mood.deltaRelax * d #mood.deltaRelax ? d : 0.0
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

function (mood::NoMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
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
function (mood::OnlyMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
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
function (mood::FirstStageNoMOOD)(particleGrid::ParticleGrid, particleIndex::Integer, rhoVec::AbstractVector{<:Real}, newRho::Real; firstStage::Bool=false)::Bool
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

struct RK3{G1 <: GradientInterpolator, G2 <: Union{GradientInterpolator, Nothing}, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}
    div2::Vector{Float64}
    function RK3(gradientInterpolator::GradientInterpolator, Nx::Integer; fallbackInterpolator::Union{GradientInterpolator,Nothing} = nothing, mood::MOODCriterion = NoMOOD())
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx))
    end
    function RK3(gradientInterpolator::GradientInterpolator, Nx::Integer, Ny::Integer; fallbackInterpolator::Union{GradientInterpolator,Nothing} = nothing, mood::MOODCriterion = NoMOOD())
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny))
    end
end

function initTimeStepper(rk3::RK3, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk4.fallbackInterpolator)
        initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end  # In case the fallbackInterpolator also starts populating the particle.alfaij fields, unpredictable things will start to happen.
end

function (rk3::RK3)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    # Fill stage 1
    map!(particle -> particle.rho, rk3.rhoInit, particleGrid.grid)
    copyCurvatures!(particleGrid)

    # Stage 2
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk3.fallbackInterpolator)
        initTimeStep(rk3.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        rk3.div1[particleIndex] = rk3.gradientInterpolator(particleGrid, particleIndex, rk3.rhoInit, eq, settings)
        particle.rho = rk3.rhoInit[particleIndex] - rk3.div1[particleIndex]*dt/2
        if rk3.mood(particleGrid, particleIndex, rk3.rhoInit, particle.rho; firstStage=true)
            rk3.div1[particleIndex] = rk3.fallbackInterpolator(particleGrid, particleIndex, rk3.rhoInit, eq, settings; setCurvature=false)
            particle.rho = rk3.rhoInit[particleIndex] - rk3.div1[particleIndex]*dt/2  
        end
    end

    # Stage 3
    # initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    # if !isnothing(rk3.fallbackInterpolator)
    #     initTimeStep(rk3.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    # end
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
    # initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    # if !isnothing(rk3.fallbackInterpolator)
    #     initTimeStep(rk3.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    # end
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


struct RK4{G1 <: GradientInterpolator, G2 <: Union{GradientInterpolator,Nothing}, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}
    div2::Vector{Float64}
    div3::Vector{Float64}
    function RK4(gradientInterpolator::GradientInterpolator, Nx::Integer; fallbackInterpolator::Union{GradientInterpolator,Nothing} = nothing, mood::MOODCriterion = NoMOOD())
        if isnothing(fallbackInterpolator) @assert mood isa NoMOOD "No fallback interpolator can only be combined with no MOOD!" end
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx))
    end
    function RK4(gradientInterpolator::GradientInterpolator, Nx::Integer, Ny::Integer; fallbackInterpolator::Union{GradientInterpolator,Nothing} = nothing, mood::MOODCriterion = NoMOOD())
        if isnothing(fallbackInterpolator) @assert mood isa NoMOOD "No fallback interpolator can only be combined with no MOOD!" end
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny))
    end
end

function initTimeStepper(rk4::RK4, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk4.fallbackInterpolator)
        initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
end

function (rk4::RK4)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    # Fill stage 1
    map!(particle -> particle.rho, rk4.rhoInit, particleGrid.grid)    
    copyCurvatures!(particleGrid)
    # Stage 2
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk4.fallbackInterpolator)
        initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        rk4.div1[particleIndex] = rk4.gradientInterpolator(particleGrid, particleIndex, rk4.rhoInit, eq, settings)
        particle.rho = rk4.rhoInit[particleIndex] - rk4.div1[particleIndex]*dt/2
        if rk4.mood(particleGrid, particleIndex, rk4.rhoInit, particle.rho; firstStage=true)
            rk4.div1[particleIndex] = rk4.fallbackInterpolator(particleGrid, particleIndex, rk4.rhoInit, eq, settings; setCurvature=false)
            particle.rho = rk4.rhoInit[particleIndex] - rk4.div1[particleIndex]*dt/2  
        end
    end

    # Stage 3
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk4.fallbackInterpolator)
        initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
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

    # Stage 4
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk4.fallbackInterpolator)
        initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
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
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(rk4.fallbackInterpolator)
        initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
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

struct RalstonRK2{G1 <: GradientInterpolator, G2 <: Union{GradientInterpolator, Nothing}, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::Union{G2,Nothing}
    mood::MOOD
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}
    function RalstonRK2(gradientInterpolator::GradientInterpolator, Nx::Integer; fallbackInterpolator::Union{T,Nothing} = nothing, mood::MOODCriterion = NoMOOD()) where T <:GradientInterpolator
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx))
    end
    function RalstonRK2(gradientInterpolator::GradientInterpolator, Nx::Integer, Ny::Integer; fallbackInterpolator::Union{T,Nothing} = nothing, mood::MOODCriterion = NoMOOD()) where T <:GradientInterpolator
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny))
    end
end

function (ralston::RalstonRK2)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    # First stage
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(ralston.fallbackInterpolator)
        initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
    map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        ralston.div1[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings)
        particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3
        if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, particle.rho; firstStage=true)
            ralston.div1[particleIndex] = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=false)
            particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3
        end
    end

    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !isnothing(ralston.fallbackInterpolator)
        initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
    # Final stage
    map!(particle -> particle.rho, ralston.rhos, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        div = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings)
        particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div/4)
        if ralston.mood(particleGrid, particleIndex, ralston.rhos, particle.rho)
            div = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings; setCurvature=false)
            particle.rho = ralston.rhos[particleIndex] - dt*div/3
            #particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div/4)
        end
    end
end

# # Alternative MOOD implementation with full RK1 fallback
# function (ralston::RalstonRK2)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
#     # First stage
#     map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid)
#     copyCurvatures!(particleGrid)
#     for (particleIndex, particle) in enumerate(particleGrid.grid)
#         ralston.div1[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings)
#         particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3
#     end
#     map!(particle -> particle.rho, ralston.rhos, particleGrid.grid)
#     copyCurvatures!(particleGrid)
#     for (particleIndex, particle) in enumerate(particleGrid.grid)
#         div = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings)
#         particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div/4)
#     end

#     for (particleIndex, particle) in enumerate(particleGrid.grid)
#         if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, particle.rho; firstStage = true)
#             div = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=false)
#             particle.rho = ralston.rhoInit[particleIndex] - dt*div
#         end
#     end
# end

struct RalstonRK2SmoothSwitch{G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}
    div_high::Vector{Float64}
    div_low::Vector{Float64}
    tol::Float64

    function RalstonRK2SmoothSwitch(gradientInterpolator::GradientInterpolator, Nx::Integer; fallbackInterpolator::GradientInterpolator = UpwindGradient(1), mood::MOODCriterion = NoMOOD(), tol::Float64 = 1.)
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), tol)
    end
    function RalstonRK2SmoothSwitch(gradientInterpolator::GradientInterpolator, Nx::Integer, Ny::Integer; fallbackInterpolator::GradientInterpolator = UpwindGradient(1; algType="Praveen"), mood::MOODCriterion = NoMOOD(), tol::Float64 = 1.) 
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), tol)
    end
end
function initTimeStepper(ralston::RalstonRK2SmoothSwitch, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)  # In case the fallbackInterpolator also starts populating the particle.alfaij fields, unpredictable things will start to happen.
end
function (ralston::RalstonRK2SmoothSwitch)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    # # First stage
    # map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid)
    # copyCurvatures!(particleGrid)
    # for (particleIndex, particle) in enumerate(particleGrid.grid)
    #     ralston.div1[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings)
    #     particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3
    # end
    # map!(particle -> particle.rho, ralston.rhos, particleGrid.grid)
    # copyCurvatures!(particleGrid)
    # #mood_indices = []
    # for (particleIndex, particle) in enumerate(particleGrid.grid)
    #     div = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings)
    #     particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div/4)
    #     ralston.div_high[particleIndex] = 2/3 * ralston.div1[particleIndex] + 1/3 * div
    # end
        # First stage
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)

    map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid)
    copyCurvatures!(particleGrid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        ralston.div1[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings)
        particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3
    end
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)

    # Final stage
    map!(particle -> particle.rho, ralston.rhos, particleGrid.grid)
    copyCurvatures!(particleGrid)
    initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        div = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings)
        particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div/4)
        ralston.div_high[particleIndex] = 2/3 * ralston.div1[particleIndex] + 1/3 *div
    end
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        div = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings)
        ralston.rhos[particleIndex] = ralston.rhoInit[particleIndex] - dt * div
        ralston.div_low[particleIndex] = div
    end
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        low = ralston.div_low[particleIndex]
        high = ralston.div_high[particleIndex]
        denom = max(abs(low), abs(high))
        if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, particle.rho; firstStage = true) || ((denom >= 10. ^-12) && (abs(high-low)/denom >= ralston.mood.tol))
            particle.rho = ralston.rhos[particleIndex]
        end
    end
end

struct RalstonRK2SmoothSwitch2{G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    rhosFB::Vector{Float64}
    div1::Vector{Float64}
    mood_indices::Vector{Int64}
    prop_indices::Vector{Int64}
    tol::Float64

    function RalstonRK2SmoothSwitch2(gradientInterpolator::GradientInterpolator, Nx::Integer; fallbackInterpolator::GradientInterpolator = gradientInterpolator, mood::MOODCriterion = NoMOOD(), tol = 10. ^-7)
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Int64}(undef, 0), Vector{Int64}(undef, 0), tol)
    end
    function RalstonRK2SmoothSwitch2(gradientInterpolator::GradientInterpolator, Nx::Integer, Ny::Integer; fallbackInterpolator::GradientInterpolator = gradientInterpolator, mood::MOODCriterion = NoMOOD(), tol = 10. ^-7) 
        new{typeof(gradientInterpolator), typeof(fallbackInterpolator), typeof(mood)}(gradientInterpolator, fallbackInterpolator, mood, Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Int64}(undef, Nx), Vector{Int64}(undef, 0), tol)
    end
end
function initTimeStepper(ralston::RalstonRK2SmoothSwitch2, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)  # In case the fallbackInterpolator also starts populating the particle.alfaij fields, unpredictable things will start to happen.
end

function (ralston::RalstonRK2SmoothSwitch2)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    
    
    target_mass = 0.
    current_mass = 0.
    empty!(ralston.mood_indices)
    empty!(ralston.prop_indices)
    map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid)
    copyCurvatures!(particleGrid)
    # Determine full fallback
    initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for (particleIndex,particle) in enumerate(particleGrid.grid)
        div = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature = false)
        rho_tmp = ralston.rhoInit[particleIndex] - div*dt
        target_mass += rho_tmp * particle.volume
        ralston.rhosFB[particleIndex] = rho_tmp
    end    

    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    # regular Ralston with MOOD, Fallback to full fallback gradient
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        ralston.div1[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings)
        particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3
    end
    map!(particle -> particle.rho, ralston.rhos, particleGrid.grid)
    copyCurvatures!(particleGrid)
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        div = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings)
        particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div/4)
        if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, particle.rho; firstStage = true)
            particle.rho = ralston.rhosFB[particleIndex]
            #append!(ralston.mood_indices, particle.neighbourIndices)
            push!(ralston.mood_indices,particleIndex)
            ralston.div1[particleIndex] = 2/3 * ralston.div1[particleIndex] + 1/3 * div
        end
        current_mass += particle.rho * particle.volume
    end
    
    for i = sortperm(ralston.div1[ralston.mood_indices])
        append!(ralston.prop_indices, particleGrid.grid[ralston.mood_indices[i]].neighbourIndices)
    end
    while abs(target_mass - current_mass) >= ralston.tol
        if isempty(ralston.prop_indices)
            @warn "Total Fallback or no MOOD detected! Mass change is " * string(abs(target_mass -current_mass))
            break
        end
        particleIndex = popfirst!(ralston.prop_indices)
        particle = particleGrid.grid[particleIndex]
        if particle.moodEvent
            continue
        end
        particle.moodEvent = true
        append!(ralston.prop_indices, particle.neighbourIndices)
        local_mass = particle.rho * particle.volume
        particle.rho = ralston.rhosFB[particleIndex]
        current_mass += particle.rho * particle.volume - local_mass

    end
end
# # Assume RalstonRK2SmoothSwitch is the type, though function signature uses RalstonRK2
# function (ralston::RalstonRK2SmoothSwitch)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
#     # --- Stage 1: Calculate k1 and intermediate u* ---
#     map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid) # Store u_n
#     copyCurvatures!(particleGrid) # For MOOD check or high-order interpolator based on u_n

#     for (particleIndex, particle) in enumerate(particleGrid.grid)
#         # k1 = gradientInterpolator(u_n)
#         # Assuming gradientInterpolator sets particle.curvature for its own needs or for MOODu2
#         ralston.div1[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=true)
#         # u* = u_n - (2/3)*dt*k1
#         particle.rho = ralston.rhoInit[particleIndex] - ralston.div1[particleIndex]*dt*2/3 # particle.rho now holds u*
#     end

#     # --- Stage 2: Calculate k2 and final high-order candidate u_n+1 ---
#     map!(particle -> particle.rho, ralston.rhos, particleGrid.grid) # Store u* from particle.rho into ralston.rhos
#     copyCurvatures!(particleGrid) # Update curvatures based on u* (ralston.rhos)

#     mood_indices = [] # Stores indices of particles that initially trigger MOOD
#     # This will store the k2_high for propagation check
#     k2_high_order_values = similar(ralston.div1) # Or add this as a field to ralston struct

#     for (particleIndex, particle) in enumerate(particleGrid.grid)
#         # k2 = gradientInterpolator(u*)
#         div_k2 = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings; setCurvature=true)
#         k2_high_order_values[particleIndex] = div_k2 # Store k2_high

#         # Final high-order update candidate: u_n+1 = u_n - dt * ( (1/4)*k1 + (3/4)*k2 )
#         # ralston.div1[particleIndex] still holds k1 from Stage 1 for this particle
#         particle.rho = ralston.rhoInit[particleIndex] - dt*(ralston.div1[particleIndex]/4 + 3*div_k2/4) # particle.rho now holds u_n+1_candidate

#         # Initial MOOD check on the final high-order candidate u_n+1 against initial state u_n
#         if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, particle.rho; firstStage = true)
#             # The ralston.mood function itself likely sets particle.moodEvent = true
#             push!(mood_indices, particleIndex)
#         end
#     end

#     # --- Iterative Fallback and Propagation Loop ---
#     # `tol` should be defined, e.g., from settings or as a constant
#     # tol = get(settings.properties, :flux_propagation_tol, 1e-4) # Example
#     # You mentioned ralston.mood.tol, which implies tol is part of the mood struct. This is good.
#     # Let's assume ralston.mood.tol exists and is accessible.
#     tol = ralston.mood.tol # Make sure 'tol' is defined in your MOOD struct if used like this

#     # Keep track of particles that have been processed by fallback in this time step
#     fallbacked_this_step = falses(length(particleGrid.grid))

#     # Process initial MOOD indices first without propagation check on them
#     initial_mood_processing_queue = copy(mood_indices) # Process these first
#     empty!(mood_indices) # This will be the queue for propagation

#     while !isempty(initial_mood_processing_queue)
#         particleIndex = popfirst!(initial_mood_processing_queue) # FIFO for initial processing

#         if fallbacked_this_step[particleIndex]
#             continue
#         end

#         # Apply fallback for this particleIndex
#         div_fallback = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=false)
#         particleGrid.grid[particleIndex].rho = ralston.rhoInit[particleIndex] - dt*div_fallback
        
#         particleGrid.grid[particleIndex].moodEvent = true # Mark that this particle *is* now MOOD
#         fallbacked_this_step[particleIndex] = true     # Mark as processed by fallback

#         # Now, check its neighbors for propagation based on this fallback
#         for nbIndex in particleGrid.grid[particleIndex].neighbourIndices
#             nbParticle = particleGrid.grid[nbIndex]
#             if !fallbacked_this_step[nbIndex] && !nbParticle.moodEvent # Only consider propagating to non-fallbacked, non-primarily-MOOD neighbors
#                 # div_fallback is the low-order tendency of particleIndex (from u_n)
#                 # k2_high_order_values[nbIndex] is the high-order k2 tendency of nbIndex (from u*)
#                 denom = max(abs(div_fallback), abs(k2_high_order_values[nbIndex]))
#                 denom = (denom ≈ 0.) ? 1 : denom
#                 if (abs(div_fallback - k2_high_order_values[nbIndex])/denom) >= tol
#                     # This neighbor is now considered for fallback due to propagation
#                     # We don't set its moodEvent here, only add to queue.
#                     # It will be set when it's processed.
#                     if !in(nbIndex, mood_indices) # Avoid duplicates in propagation queue
#                         push!(mood_indices, nbIndex) # Add to propagation queue
#                     end
#                 end
#             end
#         end
#     end

#     # Now process the propagation queue (mood_indices now only contains propagated candidates)
#     idx_prop = 0
#     while !isempty(mood_indices) && idx_prop < length(particleGrid.grid) # Safety break for propagation
#         idx_prop += 1
#         particleIndex = popfirst!(mood_indices) # FIFO can be better for layer-by-layer propagation

#         if fallbacked_this_step[particleIndex] # Could have been added multiple times before processing
#             continue
#         end

#         # Apply fallback for this propagated particleIndex
#         div_fallback = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=false)
#         particleGrid.grid[particleIndex].rho = ralston.rhoInit[particleIndex] - dt*div_fallback
        
#         particleGrid.grid[particleIndex].moodEvent = true # Mark that this particle *is* now MOOD
#         fallbacked_this_step[particleIndex] = true     # Mark as processed by fallback

#         # Check ITS neighbors for further propagation
#         for nbIndex in particleGrid.grid[particleIndex].neighbourIndices
#             nbParticle = particleGrid.grid[nbIndex]
#             if !fallbacked_this_step[nbIndex] && !nbParticle.moodEvent
#                 if abs(div_fallback - k2_high_order_values[nbIndex]) >= tol
#                     if !in(nbIndex, mood_indices) # Avoid duplicates
#                         push!(mood_indices, nbIndex)
#                     end
#                 end
#             end
#         end
#     end

#     # Reset moodEvent flags for all particles for the next time step
#     for particle_obj in particleGrid.grid # Renamed to avoid conflict
#         particle_obj.moodEvent = false
#     end
# end

# Gemini's function:
# In MeshfreeTimeSteppers.txt module TimeIntegration

# (Keep the original RalstonRK2 struct definition with fallbackInterpolator and mood)

# Replace the function (ralston::RalstonRK2)(...) with this conservative MOOD version:
# function (ralston::RalstonRK2)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
#     N = length(particleGrid.grid)
#     # Temporary storage for high-order tendencies at each stage
#     k1_high = Vector{Float64}(undef, N)
#     k2_high = Vector{Float64}(undef, N)
#     # Reuse existing field ralston.div1 to store k1_actual
#     k1_actual = ralston.div1
#     # Need storage for k2_actual
#     k2_actual = Vector{Float64}(undef, N)

#     # Store initial solution u_n
#     map!(particle -> particle.rho, ralston.rhoInit, particleGrid.grid)

#     # --- Stage 1: Determine k1_actual ---

#     # Ensure curvatures are available if needed by MOOD (assuming gradientInterpolator sets them)
#     # This loop might be optimizable if curvature setting can be done differently.
#     for particleIndex in 1:N
#          _ = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=true)
#     end
#     copyCurvatures!(particleGrid) # Make curvatures available in particleGrid.temp

#     # Calculate high-order k1 and check MOOD to determine k1_actual
#     for particleIndex in 1:N
#         k1_high[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=true)
#         rho_candidate_stage1 = ralston.rhoInit[particleIndex] - (2.0/3.0)*dt*k1_high[particleIndex]

#         # === Enforcement Point 1 ===
#         if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, rho_candidate_stage1; firstStage=true)
#             # Use fallback if MOOD triggers
#             k1_actual[particleIndex] = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhoInit, eq, settings; setCurvature=false)
#         else
#             # Use high-order otherwise
#             k1_actual[particleIndex] = k1_high[particleIndex]
#         end
#     end

#     # Compute intermediate solution u* using k1_actual
#     # ralston.rhos stores u* = u_n - (2/3)*dt*k1_actual
#     for particleIndex in 1:N
#         ralston.rhos[particleIndex] = ralston.rhoInit[particleIndex] - (2.0/3.0)*dt*k1_actual[particleIndex]
#     end

#     # --- Stage 2: Determine k2_actual ---

#     # Ensure curvatures based on u* (ralston.rhos) are available if needed
#     for particleIndex in 1:N
#          _ = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings; setCurvature=true)
#     end
#     copyCurvatures!(particleGrid)

#     # Calculate high-order k2 and check MOOD to determine k2_actual
#     for particleIndex in 1:N
#         k2_high[particleIndex] = ralston.gradientInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings; setCurvature=true)
#         rho_candidate_final = ralston.rhoInit[particleIndex] - dt * ( (1.0/4.0)*k1_actual[particleIndex] + (3.0/4.0)*k2_high[particleIndex] )

#         # === Enforcement Point 2 ===
#         # Check final candidate against initial state (or intermediate state u* depending on MOOD definition)
#         if ralston.mood(particleGrid, particleIndex, ralston.rhoInit, rho_candidate_final; firstStage=false)
#         # Alternative check: if ralston.mood(particleGrid, particleIndex, ralston.rhos, rho_candidate_final; firstStage=false)
#             # Use fallback if MOOD triggers
#             k2_actual[particleIndex] = ralston.fallbackInterpolator(particleGrid, particleIndex, ralston.rhos, eq, settings; setCurvature=false)
#         else
#             # Use high-order otherwise
#             k2_actual[particleIndex] = k2_high[particleIndex]
#         end
#     end

#     # --- Final Update ---
#     # Combine using the determined k1_actual and k2_actual
#     # u_n+1 = u_n - dt * ( (1/4)*k1_actual + (3/4)*k2_actual )
#     for particleIndex in 1:N
#         particleGrid.grid[particleIndex].rho = ralston.rhoInit[particleIndex] - dt * ( (1.0/4.0)*k1_actual[particleIndex] + (3.0/4.0)*k2_actual[particleIndex] )
#     end
# end


# In MeshfreeTimeSteppers.txt module TimeIntegration

# (Keep existing using statements and struct definitions like RalstonRK2, RK3, RK4, MOOD criteria, EulerUpwind etc.)
# ...


# --- Deprecated! regular RK2 can be used now ---

# This struct is designed to work with GradientInterpolators like MUSCLlimited
# whose initTimeStep method requires the current solution vector `fVec`.
struct RalstonRK2Limiter{G1 <: GradientInterpolator} <: MeshfreeTimeStepper
    gradientInterpolator::G1 # Should be MUSCLlimited or similar
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}    # Stores u* (intermediate stage solution)
    div1::Vector{Float64}     # Stores k1 (tendency from stage 1)

    # Constructor - Takes the GradientInterpolator (e.g., MUSCLlimited instance)
    # Assumes Nx is for 1D grid size. Add Nx, Ny constructor if needed for 2D.
    function RalstonRK2Limiter(gradientInterpolator::G1, Nx::Integer) where {G1 <: GradientInterpolator}
        # Might want to add checks here ensure G1 has the appropriate initTimeStep signature if possible
        new{G1}(gradientInterpolator, Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx))
    end
    # Add 2D constructor if necessary
    # function RalstonRK2Limiter(gradientInterpolator::G1, Nx::Integer, Ny::Integer) where {G1 <: GradientInterpolator}
    #    new{G1}(gradientInterpolator, Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny), Vector{Float64}(undef, Nx*Ny))
    # end
end

# initTimeStepper for RalstonRK2Limiter - delegates to the contained interpolator's initTimeStep
# but CANNOT pass fVec here as per the constraint. The interpolator's initTimeStep
# will be called with fVec inside the functor below.
# This function might only do geometry-based setup if the interpolator needs it
# separate from the fVec-dependent part. If MUSCLlimited's initTimeStep does everything,
# this might become a no-op or just call the geometry part if separated.
# For now, let's assume it might call the original geometry-only part if available,
# otherwise does nothing here. If MUSCLlimited ONLY has the combined initTimeStep,
# then this method should probably do nothing.
function initTimeStepper(limiter_rk2::RalstonRK2Limiter, particleGrid::ParticleGrid, settings::SimSetting)
    # If the gradientInterpolator has a separate geometry-only init method, call it here.
    # Otherwise, do nothing, as the fVec-dependent init is called inside the functor.
    # Example: if typeof(limiter_rk2.gradientInterpolator) has a method initTimeStep_geometry(...)
    #     initTimeStep_geometry(limiter_rk2.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    # end
    # Assuming the MUSCLlimited initTimeStep defined above handles both geometry (alfaij)
    # and slope limiting, this outer initTimeStep might not be strictly needed unless
    # other interpolators used with this timestepper need it.
    # For safety, let's keep it potentially calling the standard initTimeStep,
    # which MUSCLlimited won't have, so it effectively does nothing for MUSCLlimited.
    try
        # Call the standard initTimeStep which MUSCLlimited doesn't have
        # This line will likely error for MUSCLlimited, so wrap in try-catch or remove
        # initTimeStep(limiter_rk2.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    catch e
        if !(e isa MethodError) rethrow(e) end # Ignore only MethodError
    end
    # The crucial initTimeStep call happens *inside* the functor below.
end


# Functor for RalstonRK2Limiter
function (limiter_rk2::RalstonRK2Limiter)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = length(particleGrid.grid)
    # Temporary storage for k2 (tendency from stage 2)
    div2 = Vector{Float64}(undef, N)

    # Store initial solution u_n
    map!(particle -> particle.rho, limiter_rk2.rhoInit, particleGrid.grid)

    # --- Stage 1: Calculate k1 ---

    # Call initTimeStep for the gradient interpolator WITH the current solution state (rhoInit)
    # This computes limited slopes based on u_n and stores them in the interpolator's cache
    # It also computes geometric coefficients like alfaij if not already done.
    initTimeStep(limiter_rk2.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange, limiter_rk2.rhoInit)

    # Compute k1 using the interpolator (which now uses the pre-computed limited slopes)
    # Use Threads.@threads for potential parallelism if gradient calculation is independent enough
    Threads.@threads for particleIndex in 1:N
        # The functor call uses the limited_slopes cached by initTimeStep
        limiter_rk2.div1[particleIndex] = limiter_rk2.gradientInterpolator(particleGrid, particleIndex, limiter_rk2.rhoInit, eq, settings)
    end

    # Compute intermediate solution u*
    # u* = u_n - (2/3)*dt*k1
    Threads.@threads for particleIndex in 1:N
        limiter_rk2.rhos[particleIndex] = limiter_rk2.rhoInit[particleIndex] - (2.0/3.0)*dt*limiter_rk2.div1[particleIndex]
    end

    # --- Stage 2: Calculate k2 ---

    # Call initTimeStep again for the gradient interpolator WITH the intermediate state (rhos)
    # This re-computes limited slopes based on u* and stores them
    initTimeStep(limiter_rk2.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange, limiter_rk2.rhos)

    # Compute k2 using the interpolator (which now uses the limited slopes based on u*)
    Threads.@threads for particleIndex in 1:N
        # The functor call uses the new limited_slopes cached by the second initTimeStep call
        div2[particleIndex] = limiter_rk2.gradientInterpolator(particleGrid, particleIndex, limiter_rk2.rhos, eq, settings)
    end

    # --- Final Update ---
    # u_n+1 = u_n - dt * ( (1/4)*k1 + (3/4)*k2 )
    Threads.@threads for particleIndex in 1:N
        particleGrid.grid[particleIndex].rho = limiter_rk2.rhoInit[particleIndex] - dt * ( (1.0/4.0)*limiter_rk2.div1[particleIndex] + (3.0/4.0)*div2[particleIndex] )
    end
end
