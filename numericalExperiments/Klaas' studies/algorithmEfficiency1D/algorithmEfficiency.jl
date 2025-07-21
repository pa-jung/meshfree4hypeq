using FileIO
using JLD2
using Statistics
using Plots
using Dierckx
using LinearAlgebra
using LaTeXStrings
using Random
using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq

function smoothInit(x::Real)
    return exp(-x^2)
end

function shockInit(x::Real)
    return x > 0.0 ? 1.0 : 0.0
end

function mapGrid(x; xmin, xmax)
    temp = x .- xmin
    temp1 = mod.(temp, xmax-xmin)
    return temp1 .+ xmin
end

function computeError(particleGrid::ParticleGrid1D, initFunc::Function, time::Real, vel::Real; normP = 2)
    # Compute exact solution
    xsRegular = particleGrid.xmin:particleGrid.dx:particleGrid.xmax
    xsRegularT = xsRegular .- vel*time  
    xsRegularT = mapGrid(xsRegularT; xmin=particleGrid.xmin, xmax=particleGrid.xmax)
    exactSolT = initFunc.(xsRegularT)  # Exact solution at t = time

    # Interpolate solution on regular grid
    rhos = collect(map(particle -> particle.rho, particleGrid.grid))
    xsIrregular = collect(map(particle -> particle.pos, particleGrid.grid))
    push!(rhos, rhos[1])  # Add point at right boundary explicitly to array
    push!(xsIrregular, particleGrid.xmax)

    rhoSpline = Spline1D(xsIrregular, rhos; k=5, s=0.0, periodic=true)
    numericalSol = evaluate(rhoSpline, xsRegular)

    # Return absolute error
    return norm(exactSolT - numericalSol, normP)/norm(exactSolT)
end

const global a = 1.0;

function doSimluation(initFunc::String, AlgID::Integer, N::Integer, tmax::Real, repeat::Integer)
    saveDir = "$(@__DIR__)/data/"
    xmin = -5
    xmax = 5
    interpAlpha = 1.0
    interpRangeConst = 3.5

    # Equation
    eq = LinearAdvection(a)

    # Set state of rng
    Random.seed!(Meshfree4ScalarEq.rng, repeat+10)

    # Grid stuff
    dx = (xmax-xmin)/N
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness=dx*0.45)

    dtEuler = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*particleGrid.dx)

    # Simlation algorithms
    if AlgID == 1
        method = EulerUpwind(N)
        saveDir *= "MeshfreeUpwind1"
        dt = dtEuler*0.99  # CFL 1.0
    elseif AlgID == 2
        method = RK3(WENO(2), N)
        saveDir *= "RK3WENO2"
        dt = dtEuler*0.95  # idk
    elseif AlgID == 3
        method = RalstonRK2(WENO(2), N)
        saveDir *= "RalstonWENO2"
        dt = dtEuler*0.7  # 0.45 unstable
    elseif AlgID == 4
        method = RalstonRK2(UpwindGradient(2), N; mood = MOODu2(deltaRelax=true))
        saveDir *= "RalstonRK2MOODUpwind2"
        dt = dtEuler*0.3  # 0.4 unstable
    elseif AlgID == 5
        method = RalstonRK2(MUSCL(2), N; mood = MOODu2(deltaRelax=true))
        saveDir *= "RalstonRK2MOODMUSCL2"
        dt = dtEuler*0.75  # 0.8 done
    elseif AlgID == 6
        method = RK4(MUSCL(4), N; mood = MOODu2(deltaRelax=true))
        saveDir *= "RK4MOODMUSCL4"
        dt = dtEuler*0.70  # 0.70 unstable
    elseif AlgID == 7
        method = RalstonRK2(MUSCL(2), N)
        saveDir *= "RalstonRK2MUSCL2"
        dt = dtEuler*0.75
    elseif AlgID == 8
        method = RK4(MUSCL(4), N)
        saveDir *= "RK4MUSCL4"
        dt = dtEuler*0.70  
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
    saveFreq = 100000000000
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRangeConst*particleGrid.dx,
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq)

    # Do simulation
    time = mainTimeIntegrator!(method, eq, particleGrid, settings)

    # Return RNG state
    return particleGrid, settings, time
end

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

    tmax = 7.5
    Ns = [convert(Int64, round(10^x)) for x in range(log10(30), log10(1500), 10)]
    Algs = [1; 3; 4; 5; 6; 7; 8]
    AlgNames = ["MeshfreeUpwind1" "RK2 WENO2" "RK2MOOD Upwind2" "RK2MOOD MUSCL2" "RK4MOOD MUSCL4" "RK2 MUSCL2" "RK4 MUSCL4"]
    errors = Matrix{Float64}(undef, length(Ns), length(Algs))
    timings = Matrix{Float64}(undef, size(errors))
    for (i, Alg) in enumerate(Algs)
        for (j, N) in enumerate(Ns)
            try
                _, _, _ = doSimluation(initString, Alg, N, tmax, 0)  # Once for compilation
            catch e
                println(e*" for algorithm $(Alg) and Nx=$(N).")
                error("Stop here")
            end

            SimRepeats = 10

            eN = Vector{Float64}(undef, SimRepeats)
            tN = Vector{Float64}(undef, SimRepeats)

            for k = 1:SimRepeats
                particleGrid, settings, time = doSimluation(initString, Alg, N, tmax, k)
                if Alg == 3
                    println(maximum(map(particle -> particle.rho, particleGrid.grid)))
                end
                eN[k] = computeError(particleGrid, initFunc, tmax, a; normP = 2)
                tN[k] = time

                # Plot solutions with exact solution
                if k == SimRepeats
                    p = plotDensity(settings, settings.currentSaveNb-1; saveFigure=true)
                    xsRegular = particleGrid.xmin:particleGrid.dx:particleGrid.xmax
                    xsRegularT = xsRegular .- a*tmax  
                    xsRegularT = mapGrid(xsRegularT; xmin=particleGrid.xmin, xmax=particleGrid.xmax)
                    exactSolT = initFunc.(xsRegularT)  # Exact solution at t = tmax
                
                    plot!(p, xsRegular, exactSolT, label="Exact solution")
                    savefig(p, "$(settings.saveDir)figures/densityPlot$(settings.currentSaveNb-1).png")
                end
            end
            errors[j, i] = mean(eN)
            timings[j, i] = mean(tN)
            println("Algorithm $(Alg) for Nx=$(N) done.")
        end
    end

    save("$(@__DIR__)/data/efficiencyResults$(init).jld2", "timings", timings, "errors", errors, "AlgNames", AlgNames)
end

checkError()