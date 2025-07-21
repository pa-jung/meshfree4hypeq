using ProgressMeter
using Plots
using Random
using LinearAlgebra
using LaTeXStrings
using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq
gr()

function smoothInit1(x::Real)
    return exp(-x^2)
end

function smoothInit2(x::Real)
    return sin(2*π*x/5)
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
    xsIrregular = collect(map(particle -> particle.pos, particleGrid.grid))
    numericalSol = collect(map(particle -> particle.rho, particleGrid.grid))
    exactSol = [initFunc(mapGrid(x - vel*time, xmin=particleGrid.xmin, xmax=particleGrid.xmax)) for x in xsIrregular]
    return norm(exactSol - numericalSol, normP)/norm(exactSol, normP)
end

const global a = 1.0;

function doSimluation(initFunc::String, AlgID::Integer, N::Integer, tmax::Real)
    saveDir = "$(@__DIR__)/data/"
    xmin = -5
    xmax = 5
    interpAlpha = 1.0
    interpRangeConst = 3.5

    # Equation
    eq = LinearAdvection(a)

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Grid stuff
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness=(xmax-xmin)/(2*N))

    dtEuler = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*particleGrid.dx)

    # Simlation algorithms
    if AlgID == 1
        method = EulerUpwind(N)
        saveDir *= "MeshfreeUpwind1"
    elseif AlgID == 2
        method = RK3(CentralGradient(1), N)
        saveDir *= "RK3Central1"
    elseif AlgID == 3
        method = RK3(CentralGradient(1), N; mood = MOODu2(deltaRelax=true))
        saveDir *= "RK3MOODCentral1"
    elseif AlgID == 4
        method = RK3(UpwindGradient(1), N)
        saveDir *= "RK3Upwind1"
    elseif AlgID == 5
        method = RK3(CentralGradient(2), N)
        saveDir *= "RK3Central2"
    elseif AlgID == 6
        method = RK3(CentralGradient(2), N; mood = MOODu2(deltaRelax=true))
        saveDir *= "RK3MOODCentral2"
    elseif AlgID == 7
        method = RK3(UpwindGradient(2), N)
        saveDir *= "RK3Upwind2"
    elseif AlgID == 8
        method = RK3(WENO(2), N)
        saveDir *= "RK3WENO2"
    elseif AlgID == 9
        method = RalstonRK2(WENO(2), N)
        saveDir *= "RalstonWENO2"
    elseif AlgID == 10
        method = RalstonRK2(CentralGradient(2), N)
        saveDir *= "RalstonCentralGradient2"
    elseif AlgID == 11
        method = RK3(UpwindGradient(2), N; mood = MOODu2(deltaRelax=true))
        saveDir *= "RK3MOODUpwind2"
    elseif AlgID == 12
        method = RK3(MUSCL(1), N)  # MUSCL with linear reconstruction
        saveDir *= "RK3MUSCL1"  
    elseif AlgID == 13
        method = RK3(MUSCL(2), N)  # MUSCL with quadratic reconstruction
        saveDir *= "RK3MUSCL2"
    elseif AlgID == 14
        method = RK3(MUSCL(3), N)  # MUSCL with cubic reconstruction
        saveDir *= "RK3MUSCL3"
    elseif AlgID == 15
        method = RK3(MUSCL(4), N)  # MUSCL with quartic reconstruction
        saveDir *= "RK3MUSCL4"
    elseif AlgID == 16
        method = RK3(MUSCL(2), N; mood = MOODu2(deltaRelax=true))  # MUSCL with quadratic reconstruction
        saveDir *= "RK3MOODMUSCL2"
    elseif AlgID == 17
        method = RK3(MUSCL(4), N; mood = MOODu2(deltaRelax=true))  # MUSCL with quartic reconstruction
        saveDir *= "RK3MOODMUSCL4"
    elseif AlgID == 18
        method = RK3(MUSCL(4), N; mood = MOODu1(deltaRelax=true))  # MUSCL with quartic reconstruction
        saveDir *= "RK3MOODu1MUSCL4"
    end
    saveDir *= "_irgrid"

    dt = dtEuler/20

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

    if hasproperty(method, :mood) && (initFunc == "smoothInit1" || initFunc == "smoothInit2")
        println("$(settings.saveDir) moodRatio: $(method.mood.count/(N*tmax/dt))")
    end

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
    return particleGrid, settings, time
end

function checkError(init::Integer)

    if init == 1
        initString = "smoothInit1"
        initFunc = smoothInit1
    elseif init == 2  
        initString = "shockInit"
        initFunc = shockInit
    elseif init == 3
        initString = "smoothInit2"
        initFunc = smoothInit2
    else
        error("Input error.")
    end

    tmax = 2.5
    Ns = [30; 50; 70; 100; 175; 420; 700; 1000]
    Algs = [1; 4; 7; 8; 11; 12; 13; 15; 16; 17; 9; 18]
    AlgNames = ["MeshfreeUpwind1" "RK3Upwind1" "RK3Upwind2" "RK3WENO2" "RK3MOODUpwind2" "RK3 MUSCL1" "RK3 MUSCL2" "RK3 MUSCL4" "RK3MOOD MUSCL2" "RK3MOOD MUSCL4" "RK2WENO2" "RK3MOODu1 MUSCL4"]
    AlgNamesClean = ["MeshfreeUpwind1"; "Upwind: order 1"; "Upwind: order 2"; "WENO: order 2"; "Upwind + MOOD: order 2"; "MUSCL: order 1"; "MUSCL: order 2"; "MUSCL: order 4"; "MUSCL + MOOD: order 2"; "MUSCL + MOOD: order 4"; "RK2 WENO2"; "MUSCL + MOODu1: order 4"]
    
    errors = Matrix{Float64}(undef, length(Ns), length(Algs))
    for (i, Alg) in enumerate(Algs)
        for (j, N) in enumerate(Ns)
            particleGrid, settings, _ = doSimluation(initString, Alg, N, tmax)
            errors[j, i] = computeError(particleGrid, initFunc, tmax, a; normP = 2)

            # Plot solutions with exact solution
            p = plotDensity(settings, settings.currentSaveNb-1; saveFigure=false, showMOODEvents=true)
            xsRegular = particleGrid.xmin:particleGrid.dx:particleGrid.xmax
            xsRegularT = xsRegular .- a*tmax  
            xsRegularT = mapGrid(xsRegularT; xmin=particleGrid.xmin, xmax=particleGrid.xmax)
            exactSolT = initFunc.(xsRegularT)  # Exact solution at t = tmax
        
            plot!(p, xsRegular, exactSolT, label="Exact solution")
            savefig(p, "$(settings.saveDir)figures/densityPlot$(settings.currentSaveNb-1).png")
        end
    end

    # Plot convergence
    p1 = plot(Ns, errors, xscale=:log10, yscale=:log10, label=AlgNames, legend=:bottomleft, ylabel="Relative error", xlabel=L"N_x", title="Global error at t=$(tmax)s", ls=:solid, markershape=:circle, markersize=3)
    plot!(p1, Ns, 10 ./ Ns, label="Reference order 1", ls=:dash)
    plot!(p1, Ns, 50 ./ (Ns.^2), label="Reference order 2", ls=:dash)
    plot!(p1, Ns, 100 ./ (Ns.^3), label="Reference order 3", ls=:dash)
    plot!(p1, Ns, 1000 ./ (Ns.^4), label="Reference order 4", ls=:dash)
    savefig(p1, "$(@__DIR__)/convergence_$(initString).pdf")

    # Plot convergence for paper
    p2 = plot(Ns, errors[:, 2], xscale=:log10, yscale=:log10, label=AlgNamesClean[2], legend=:bottomleft, ylabel="Relative error", xlabel=L"N_x", ls=:dashdot, markershape=:circle, markersize=3, c=:blue)  # Upwind: order 1
    # plot!(p2, Ns, errors[:, 6], label=AlgNamesClean[6], ls=:solid, markershape=:circle, markersize=3, c=:steelblue)  # MUSCL: order 1
    plot!(p2, Ns, errors[:, 4], label=AlgNamesClean[4], ls=:dashdotdot, markershape=:circle, markersize=3, c=:green)  # WENO: order 2
    plot!(p2, Ns, errors[:, 3], label=AlgNamesClean[3], ls=:dashdot, markershape=:circle, markersize=3, c=:purple)  # Upwind: order 2
    plot!(p2, Ns, errors[:, 5], label=AlgNamesClean[5], ls=:dashdot, markershape=:star, markersize=3, c=:purple)  # Upwind + MOOD: order 2
    plot!(p2, Ns, errors[:, 7], label=AlgNamesClean[7], ls=:solid, markershape=:circle, markersize=3, c=:gold3)  # MUSCL: order 2
    plot!(p2, Ns, errors[:, 9], label=AlgNamesClean[9], ls=:solid, markershape=:star, markersize=3, c=:gold3)  # MUSCL + MOOD: order 2
    plot!(p2, Ns, errors[:, 8], label=AlgNamesClean[8], ls=:solid, markershape=:circle, markersize=3, c=:maroon1)  # MUSCL: order 4
    plot!(p2, Ns, errors[:, 10], label=AlgNamesClean[10], ls=:solid, markershape=:star, markersize=3, c=:maroon1)  # MUSCL + MOOD: order 4
    if (init == 1) || (init == 3)
        plot!(p2, Ns, 10 ./ Ns, label="Reference order 1", ls=:dash)
        plot!(p2, Ns, 50 ./ (Ns.^2), label="Reference order 2", ls=:dash)
        # plot!(p2, Ns, 100 ./ (Ns.^3), label="Reference order 3", ls=:dash)
        plot!(p2, Ns, 1000 ./ (Ns.^4), label="Reference order 4", ls=:dash)
    elseif init == 2
        plot!(p2, Ns, 1 ./ sqrt.(Ns), label="Reference order 0.5", ls=:dash)
    end
    savefig(p2, "$(@__DIR__)/convergencePaper_$(initString).pdf")
    
end

checkError(1)
#checkError(2)
#checkError(3)
