using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq

function smoothInit1(x::Real)
    return exp(-x^2)
end

function smoothInit2(x::Real)
    return sin(2*π*x/5)
end

function shockInit(x::Real)
    return x > 0.0 ? 1.0 : 0.0
end

function doSimluation(regular::Bool, initFunc::String, AlgID::Integer)
    # Simulation settings
    CFL = 1/3
    tmax = 10.0
    N = 100
    xmin = -5
    xmax = 5
    saveDir = "$(@__DIR__)/data/"
    interpAlpha = 6.0
    saveFreq = 10

    state = copy(Meshfree4ScalarEq.rng)

    # Simlation algorithms
    if AlgID == 1
        method = Upwind(N)
        saveDir *= "Upwind"
    elseif AlgID == 2
        method = EulerUpwind(N)
        saveDir *= "MeshfreeUpwind"
    elseif AlgID == 3
        method = RalstonRK2(MUSCL(1), N)
        saveDir *= "RK2MUSCL1"
    elseif AlgID == 4
        method = RK3(WENO(2), N)
        saveDir *= "RK3WENO2"
    elseif AlgID == 5
        method = RK3(MUSCL(1), N; mood = MOODu1(deltaRelax=true))
        saveDir *= "RK3MOODMUSCL1"
    elseif AlgID == 6
        method = RK3(MUSCL(2), N; mood = MOODu2(deltaRelax=true))
        saveDir *= "RK3MOODMUSCL2"
    elseif AlgID == 7
        method = RK3(MUSCL(1), N)
        saveDir *= "RK3MUSCL2"
    elseif AlgID == 8
        method = RK3(MUSCL(2), N)
        saveDir *= "RK3MUSCL2"
    elseif AlgID == 9
        method = RK3(MUSCL(3), N)
        saveDir *= "RK3MUSCL3"
    elseif AlgID == 10
        method = RK3(MUSCL(4), N; mood = NoMOOD())
        saveDir *= "RK4MUSCL4"
    elseif AlgID == 11
        method = RK3(MUSCL(4), N; mood = MOODu1(deltaRelax=true))
        saveDir *= "RK4MOODu1MUSCL4"
    elseif AlgID == 12
        method = RK3(MUSCL(4), N; mood = MOODu2(deltaRelax=true))
        saveDir *= "RK4MOODu2MUSCL4"
    end

    # Equation
    a = 1.0
    eq = LinearAdvection(a)
    
    # Grid stuff
    if regular
        particleGrid = ParticleGrid1D(xmin, xmax, N)
        saveDir *= "_rgrid"
    else
        dx = (xmax-xmin)/N
        particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = dx/2)
        saveDir *= "_irgrid"
    end
    interpRange = 3.5*particleGrid.dx
    dt = CFL*getTimeStep(particleGrid, eq, interpAlpha, interpRange)  # Time step such that meshfree first order least squares method is positive. (Check MeshfreeUpwind_irgrid_shockInit)

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

doSimluation(true, "smoothInit1", 1)
doSimluation(true, "smoothInit1", 2)
doSimluation(false, "smoothInit1", 2)
doSimluation(true, "shockInit", 1)
doSimluation(true, "shockInit", 2)
doSimluation(false, "shockInit", 2)
doSimluation(false, "smoothInit1", 3)
doSimluation(false, "shockInit", 3)
doSimluation(false, "smoothInit1", 4)
doSimluation(false, "shockInit", 4)
doSimluation(false, "smoothInit1", 5)
doSimluation(false, "shockInit", 5)
doSimluation(false, "smoothInit1", 6)
doSimluation(false, "shockInit", 6)
doSimluation(false, "smoothInit1", 7)
doSimluation(false, "shockInit", 7)
doSimluation(false, "smoothInit1", 8)
doSimluation(false, "shockInit", 8)
doSimluation(false, "smoothInit1", 9)
doSimluation(false, "shockInit", 9)
doSimluation(false, "smoothInit1", 10)
doSimluation(false, "smoothInit2", 10)
doSimluation(false, "shockInit", 10)
doSimluation(false, "smoothInit1", 11)
doSimluation(false, "shockInit", 11)
doSimluation(false, "smoothInit1", 12)
doSimluation(false, "smoothInit2", 12)
doSimluation(false, "shockInit", 12)
