using Distributed
@everywhere begin
    using JLD2
    using Plots
    using FileIO
    using Dierckx
    using LinearAlgebra
    using LaTeXStrings
    using Meshfree4ScalarEq.ScalarHyperbolicEquations
    using Meshfree4ScalarEq.ParticleGrids
    using Meshfree4ScalarEq.TimeIntegration
    using Meshfree4ScalarEq.Interpolations
    using Meshfree4ScalarEq.SimSettings
    using Meshfree4ScalarEq

function smoothInit(x::Real, y::Real)
    return exp(-x^2 - y^2)
end

function shockInit(x::Real, y::Real)
    return (-0.5 < x < 0.5) && (-0.5 < y < 0.5) ? 1.0 : 0.0
end

const global a = (1.0, 1.0);

function doSimluation(initFunc::String, AlgID::Integer, N::Integer, tmax::Real)
    saveDir = "$(@__DIR__)/data/"
    xmin = -5
    xmax = 5
    interpAlpha = 6
    interpRangeConst = sqrt(5.0^2 + 3.0^2)
    CFL = 1/40

    # Equation
    eq = LinearAdvection(a)

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Grid stuff
    dx = (xmax-xmin)/N
    dy = (xmax-xmin)/N
    particleGrid = ParticleGrid2D(xmin, xmax, xmin, xmax, N, N; randomness = (dx/2, dy/2))  
    dt = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*dx)*CFL    

    # Simlation algorithms
    if AlgID == 1
        method = EulerUpwind(N, N; algType="Praveen")
        saveDir *= "MeshfreeUpwind1"
    elseif AlgID == 2
        method = RK3(UpwindGradient(1; algType="Classic"), N, N)
        saveDir *= "RK3Upwind1"
    elseif AlgID == 3
        method = RK3(UpwindGradient(2; algType="Classic"), N, N)
        saveDir *= "RK3Upwind2"
    elseif AlgID == 4
        method = RK3(WENO(2), N, N)
        saveDir *= "RK3WENO2"
    elseif AlgID == 5
        method = RalstonRK2(WENO(2), N, N)
        saveDir *= "RalstonWENO2"
    elseif AlgID == 6
        method = RK3(UpwindGradient(2; algType="Classic"), N, N; mood=MOODu2(deltaRelax=true))
        saveDir *= "RK3MOODUpwind2"
    elseif AlgID == 7
        method = RK3(MUSCL(1), N, N)  # MUSCL with linear reconstruction
        saveDir *= "RK3MUSCL1"  
    elseif AlgID == 8
        method = RK3(MUSCL(2), N, N)  # MUSCL with quadratic reconstruction
        saveDir *= "RK3MUSCL2"
    elseif AlgID == 9
        method = RK3(MUSCL(1), N, N; mood=MOODu2(deltaRelax=true))  # MUSCL with linear reconstruction
        saveDir *= "RK3MOODMUSCL1"  
    elseif AlgID == 10
        method = RK3(MUSCL(2), N, N; mood=MOODu2(deltaRelax=true))  # MUSCL with quadratic reconstruction
        saveDir *= "RK3MOODMUSCL2"
    elseif AlgID == 11
        method = EulerUpwind(N, N; algType="NonLinearPraveen")
        saveDir *= "NonLinearPraveen"
    end

    saveDir *= "_irgrid"

    # Initial condition
    if initFunc == "smoothInit"
        setInitialConditions!(particleGrid, smoothInit)
        saveDir *= "_smoothInit"
    elseif initFunc == "shockInit"
        setInitialConditions!(particleGrid, shockInit)
        saveDir *= "_shockInit"
    else 
        error("Wrong input.")
    end

    saveDir *= "_$(N)"

    # Create SimSettings object
    saveFreq = 10000000
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRangeConst*particleGrid.dx,
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq)

    # Do simulation
    time = mainTimeIntegrator!(method, eq, particleGrid, settings)
    plotDensity(settings, settings.currentSaveNb-1; saveFigure=true)

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
    return particleGrid, settings, time
end

function computeSimulationError(tup::Tuple)
    initString, initFunc, Alg, N, tmax = tup
    _, _, _ = doSimluation(initString, Alg, N, 1e-12)  # Run once to compile all functions. Results are overwritten in second run.
    _, _, time = doSimluation(initString, Alg, N, tmax)
    println("Finished N=$(N), Alg=$(Alg) simulation.")
    return time
end

end  # @everywhere

function checkError()

    init = parse(Int64, ARGS[1])

    if init == 1
        initString = "smoothInit"
        initFunc = smoothInit
    elseif init == 2  
        initString = "shockInit"
        initFunc = shockInit
    else
        error("Input error.")
    end

    tmax = 1.0
    Ns = [30; 50; 70; 100; 175; 250]
    Algs = [11]
    
    params = [(initString, initFunc, Alg, N, tmax) for N in Ns for Alg in Algs]
    pmapRes = pmap(computeSimulationError, params)
    save("$(@__DIR__)/data/timingResults_$(initString).jld2", "pmapRes", pmapRes, "Ns", Ns, "Algs", Algs)
end

checkError()