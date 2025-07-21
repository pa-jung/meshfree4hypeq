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
    return sin(2*π*x/5) + 1.0
end

function shockInit(x::Real)
    if x > 0.0
        return 1.0
    else
        return -1.0
    end
end

function doSimluation(methodString::String, initFunc::String, randomnessFactor::Real = 1/4)
    # Simulation settings
    CFL = 1/5
    tmax = 10.0
    N = 100
    xmin = -5
    xmax = 5
    saveDir = "$(@__DIR__)/data/"
    interpAlpha = 1.0
    saveFreq = 5

    state = copy(Meshfree4ScalarEq.rng)

    # Simlation algorithms
    if methodString == "muscl2RusanovFlux"
        method = RalstonRK2(MUSCL(2; numericalFlux=RusanovFlux()), N)
        saveDir *= "muscl2RusanovFlux"
    elseif methodString == "lf"
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
    saveDir *= "_irgrid"

    interpRange = 3.5*particleGrid.dx
    dt = CFL*getTimeStep(particleGrid, eqLin, interpAlpha, interpRange)  # Time step such that meshfree first order least squares method is positive. (Check MeshfreeUpwind_irgrid_shockInit)

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
    for i = 1:10
        if i < settings.currentSaveNb-1
            plotDensity(settings, i; saveFigure=true, showMOODEvents=true)
        end
    end
    plotDensity(settings, settings.currentSaveNb-1; saveFigure=true, showMOODEvents=true)
    animateDensity(settings; saveFigure=true, fps=2, showMOODEvents=true)

    copy!(Meshfree4ScalarEq.rng, state)
end

doSimluation("EulerMUSCL1", "shockInit")
doSimluation("EulerMUSCL1", "smoothInit1")
doSimluation("EulerMUSCL1", "smoothInit2")
doSimluation("muscl2RusanovFlux", "shockInit")
doSimluation("muscl2RusanovFlux", "smoothInit1")
doSimluation("muscl2RusanovFlux", "smoothInit2")
doSimluation("muscl2UpwindFlux", "shockInit")
doSimluation("muscl2UpwindFlux", "smoothInit1")
doSimluation("muscl2UpwindFlux", "smoothInit2")
doSimluation("EulerUpwind", "shockInit")
doSimluation("EulerUpwind", "smoothInit1")
doSimluation("EulerUpwind", "smoothInit2")
doSimluation("lf", "shockInit", 0)
doSimluation("lf", "smoothInit1", 0)
doSimluation("lf", "smoothInit2", 0)
