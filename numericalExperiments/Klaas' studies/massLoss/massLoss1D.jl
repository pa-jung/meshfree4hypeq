using Distributed

@everywhere begin
    using ProgressMeter
    using Plots
    using LinearAlgebra
    using SparseArrays
    using Arpack
    using LaTeXStrings
    using Meshfree4ScalarEq.ScalarHyperbolicEquations
    using Meshfree4ScalarEq.ParticleGrids
    using Meshfree4ScalarEq.TimeIntegration
    using Meshfree4ScalarEq.Interpolations
    using Meshfree4ScalarEq.SimSettings
    using Meshfree4ScalarEq.ParticleGridStability
    using Meshfree4ScalarEq
end

@everywhere begin

function smoothInit(x::Real)
    return exp(-x^2)
end

function shockInit(x::Real)
    return x > 0.0 ? 1.0 : 0.0
end

const global N = 100;
const global xmin = -5;
const global xmax = 5;
const global a = 1.0;
const global interpAlpha = 1.0;
const global interpRangeConst = 3.5;
const global tmax = 200.0;

function doSimluation(initFunc::String, AlgID::Integer, weightFunc = exponentialWeightFunction())
    saveDir = "$(@__DIR__)/data1D/"

    # Equation
    eq = LinearAdvection(a)

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Grid stuff
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = (xmax - xmin)/(2*N))
    dtEuler = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*particleGrid.dx)

    # Simlation algorithms
    if AlgID == 1
        method = EulerUpwind(N; weightFunction=weightFunc)
        saveDir *= "MeshfreeUpwind1"
    elseif AlgID == 2
        method = RK3(CentralGradient(1; weightFunction=weightFunc), N)
        saveDir *= "RK3Central1"
    elseif AlgID == 3
        method = RK3(CentralGradient(1; weightFunction=weightFunc), N; mood=MOODu1(deltaRelax=true))
        saveDir *= "RK3MOODCentral1"
    elseif AlgID == 4
        method = RK3(UpwindGradient(1; weightFunction=weightFunc), N)
        saveDir *= "RK3Upwind1"
    elseif AlgID == 5
        method = RK3(CentralGradient(2; weightFunction=weightFunc), N)
        saveDir *= "RK3Central2"
    elseif AlgID == 6
        method = RK3(UpwindGradient(2; weightFunction=weightFunc), N)
        saveDir *= "RK3Upwind2"
    elseif AlgID == 7
        method = RK3(UpwindGradient(2; weightFunction=weightFunc), N; mood=MOODu2(deltaRelax=true))
        saveDir *= "RK3MOODUpwind2"
    elseif AlgID == 8
        method = RK3(WENO(2; weightFunction=weightFunc), N)
        saveDir *= "RK3WENO2"
    elseif AlgID == 9
        method = RalstonRK2(WENO(2; weightFunction=weightFunc), N)
        saveDir *= "RalstonWENO2"
    elseif AlgID == 10
        method = RalstonRK2(CentralGradient(2; weightFunction=weightFunc), N)
        saveDir *= "RalstonCentralGradient2"
    elseif AlgID == 11
        method = RK3(MUSCL(1; weightFunction=weightFunc), N)
        saveDir *= "RK3MUSCL1"  # MUSCL with linear reconstruction
    elseif AlgID == 12
        method = RK3(MUSCL(2; weightFunction=weightFunc), N; mood=MOODu2(deltaRelax=true))  # MUSCL with quadratic reconstruction
        saveDir *= "RK3MOODMUSCL2"
    elseif AlgID == 13
        method = RK3(MUSCL(3; weightFunction=weightFunc), N)  # MUSCL with cubic reconstruction
        saveDir *= "RK3MUSCL3"
    elseif AlgID == 14
        method = RK3(MUSCL(4; weightFunction=weightFunc), N; mood=MOODu2(deltaRelax=true))  # MUSCL with quartic reconstruction
        saveDir *= "RK3MOODMUSCL4"
    elseif AlgID == 15
        method = RK3(MUSCL(2; weightFunction=weightFunc), N)  # MUSCL with quadratic reconstruction
        saveDir *= "RK3MUSCL2"
    elseif AlgID == 16
        method = RK3(MUSCL(4; weightFunction=weightFunc), N)  # MUSCL with quartic reconstruction
        saveDir *= "RK3MUSCL4"
    end

    dt = dtEuler/4
    
    saveDir *= "_irgrid"

    # Initial condition
    if initFunc == "smoothInit"
        setInitialConditions!(particleGrid, smoothInit)
        saveDir *= "_smoothInit"
    elseif initFunc == "shockInit"
        setInitialConditions!(particleGrid, shockInit)
        saveDir *= "_shockInit"
    end

    # Create SimSettings object
    saveFreq = 50
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRangeConst*particleGrid.dx,
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq)

    # Do simulation
    mainTimeIntegrator!(method, eq, particleGrid, settings)
    plotDensity(settings, settings.currentSaveNb-1; saveFigure=true, showMOODEvents=true)
    animateDensity(settings; saveFigure=true, fps=2, showMOODEvents=true)

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
end

end  # everywhere

function runSimulations()
    grids = ["smoothInit"; ]
    algs = [4; 6; 7; 8; 11; 12; 14; 15; 16]
    simArgs = [(grid, alg) for grid in grids for alg in algs]
    for simArg in simArgs
        doSimluation(simArg[1], simArg[2])
    end
end

runSimulations()
