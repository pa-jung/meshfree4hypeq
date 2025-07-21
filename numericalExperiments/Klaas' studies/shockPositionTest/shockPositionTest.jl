using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq.FluxFunctions
using Meshfree4ScalarEq

function smoothInit1(x::Real)
    return exp(-x^2)
end

function smoothInit2(x::Real)
    return sin(2*π*x/5) + 2.0
end

function shockInit(x::Real)
    if x > 1.0
        return 0.5
    else
        return 1.5
    end
end

function doSimluation(methodString::String, initFunc::String, N::Int, randomnessFactor::Real = 1/4)
    # Simulation settings
    CFL = 1/5
    tmax = 7.5
    xmin = 0
    xmax = 10
    saveDir = "$(@__DIR__)/data/"
    interpAlpha = 1.0
    saveFreq = convert(Int64,1e10)

    state = copy(Meshfree4ScalarEq.rng)

    # Simlation algorithms
    if methodString == "muscl2RusanovFluxMOOD"
        method = RalstonRK2(MUSCL(2; numericalFlux=RusanovFlux()), N; mood=MOODu1(deltaRelax=false))
        saveDir *= "muscl2RusanovFluxMOOD"
    elseif methodString == "muscl4RusanovFluxMOOD"
        method = RalstonRK2(MUSCL(4; numericalFlux=RusanovFlux()), N; mood=MOODu1(deltaRelax=false))
        saveDir *= "muscl4RusanovFluxMOOD"
    elseif methodString == "muscl2RusanovFlux"
        method = RalstonRK2(MUSCL(2; numericalFlux=RusanovFlux()), N)
        saveDir *= "muscl2RusanovFlux"
    elseif methodString == "LF"
        method = LaxFriedrich(N)
        saveDir *= "LF"
    elseif methodString == "muscl2UpwindFlux"
        method = RalstonRK2(MUSCL(2; numericalFlux=UpwindFlux()), N)
        saveDir *= "muscl2UpwindFlux"
    elseif methodString == "EulerUpwind"
        method = EulerUpwind(N)
        saveDir *= "EulerUpwind"
    elseif methodString == "EulerMUSCL1"
        method = EulerUpwind(N; gradientInterpolator=MUSCL(1; numericalFlux=UpwindFlux()))
        saveDir *= "EulerMUSCL1"
    else
        error("Wrong method name.")
    end

    # Equation
    eq = BurgersEquation()
    eqLin = LinearAdvection(1.0)
    
    # Grid stuff
    dx = (xmax-xmin)/N
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = randomnessFactor*dx)

    interpRange = 3.5*particleGrid.dx
    dt = CFL*getTimeStep(particleGrid, LinearAdvection(1.5), interpAlpha, interpRange)  # Time step such that meshfree first order least squares method is positive. (Check MeshfreeUpwind_irgrid_shockInit)

    # Initial condition
    if initFunc == "smoothInit1"
        setInitialConditions!(particleGrid, smoothInit1)
        saveDir *= "_smoothInit1"
    elseif initFunc == "smoothInit2"
        setInitialConditions!(particleGrid, smoothInit2)
        saveDir *= "_smoothInit2"
    elseif initFunc == "shockInit"
        setInitialConditions!(particleGrid, shockInit)
        saveDir *= "_shockInit"
    end
    saveDir *= "_$(N)"

    # Create SimSettings object
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRange,
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq)

    # Do simulation
    mainTimeIntegrator!(method, eq, particleGrid, settings)

    # Generate plots
    plotDensity(settings, settings.currentSaveNb-1; saveFigure=true, showMOODEvents=true)
    # animateDensity(settings; saveFigure=true, fps=2, showMOODEvents=true)

    copy!(Meshfree4ScalarEq.rng, state)
end

# doSimluation("LF", "shockInit", 500, 0)
doSimluation("muscl2RusanovFluxMOOD", "shockInit", 250, 1/4)
doSimluation("muscl2RusanovFlux", "shockInit", 250, 1/4)
doSimluation("EulerUpwind", "shockInit", 250, 1/4)
doSimluation("muscl4RusanovFluxMOOD", "shockInit", 250, 1/4)
