using Distributed

@everywhere begin
    using Random
    using JLD2
    using FileIO
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

"""
    cleanDir(saveDir::String)

Delete all data files but the final grid.
"""
function cleanDir(saveDir::String)
    lst = readdir(saveDir*"data/")
    if length(lst) > 1
        lstNb = [parse(Int64, file[5:end-5]) for file in lst[2:end]]  # skip settings.jld2
        max = maximum(lstNb)
        for file in lst
            if file != "step$(max).jld2"
                rm(saveDir*"/data/$(file)")
            end
        end
    end
end

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

function checkStability(saveName::String, interpConstant::Real, weightFunc::MLSWeightFunction, seed::Integer) 
    saveDir = "$(@__DIR__)/data/"
    stabilityCondition = 1/8

    # Stability regions of RK2 and RK3
    RK1(z) = @. abs(1.0 + z) 
    RK2(z) = @. abs(1.0 + z + 0.5*z^2) 
    RK3(z) = @. abs(1.0 + z + 0.5*z^2 + (z^3)/6) 
    RK4(z) = @. abs(1.0 + z + 0.5*z^2 + (z^3)/6 + (z^4)/24) 

    # Generate grid
    eq = LinearAdvection(a)
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = (xmax - xmin)/(2*N), rng=MersenneTwister(seed))

    updateNeighbours!(particleGrid, interpConstant*particleGrid.dx)
    dt = stabilityCondition*getTimeStep(particleGrid, eq, interpAlpha, interpConstant*particleGrid.dx)

    A4 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "upwind", 2, weightFunc))
    lambdas4 = eigvals(A4)

    A6 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "muscl", 2, weightFunc))
    lambdas6 = eigvals(A6)

    A8 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "muscl", 4, weightFunc))
    lambdas8 = eigvals(A8)

    if typeof(weightFunc) == exponentialWeightFunction
        wString = "exp weight"
    elseif typeof(weightFunc) == inverseWeightFunction
        wString = "inv weight"
    else
        error("Weightfunction unknown")
    end

    # Plotting of eigenvalues
    ms = 4
    p1 = plot(real.(lambdas4*dt), imag.(lambdas4*dt), seriestype=:scatter, legend=false, ylabel=L"Im(\lambda dt)", xlabel=L"Re(\lambda dt)", label="Upwind: order 2", ms=ms, aspect_ratio=:equal, framestyle=:origin, markershape=:cross)
    plot!(p1, real.(lambdas6*dt), imag.(lambdas6*dt), label="MUSCL 2", seriestype=:scatter, ms=ms, markershape=:diamond)
    plot!(p1, real.(lambdas8*dt), imag.(lambdas8*dt), label="MUSCL 4 ", seriestype=:scatter, ms=ms, markershape=:diamond)
    
    # Add stability regions
    xminPlot, xmaxPlot = xlims(p1)
    yminPlot, ymaxPlot = ylims(p1)
    xs = range(xminPlot, xmaxPlot, length=200)
    ys = range(yminPlot, ymaxPlot, length=200)
    Z = xs' .+ ys*im
    plot!(p1, xs, ys, RK1(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK1 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    plot!(p1, xs, ys, RK2(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK2 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    plot!(p1, xs, ys, RK3(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK3 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    plot!(p1, xs, ys, RK4(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK4 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    
    savefig(p1, saveDir*"$(saveName)_$(wString)_$(seed).pdf")

    tol = 1e-13
    
    return [all(real.(lambdas4) .< tol); all(real.(lambdas6) .< tol); all(real.(lambdas8) .< tol)]
end


function computeStabilities(saveName::String, interpConstant::Real, weightFunc::MLSWeightFunction, randomGrid::Bool) 
    saveDir = "$(@__DIR__)/"
    stabilityCondition = 1/8

    # Stability regions of RK2 and RK3
    RK1(z) = @. abs(1.0 + z) 
    RK2(z) = @. abs(1.0 + z + 0.5*z^2) 
    RK3(z) = @. abs(1.0 + z + 0.5*z^2 + (z^3)/6) 
    RK4(z) = @. abs(1.0 + z + 0.5*z^2 + (z^3)/6 + (z^4)/24) 

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Generate grid
    eq = LinearAdvection(a)
    if randomGrid
        particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = (xmax - xmin)/(2*N))
    else
        particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = 0)
    end
    updateNeighbours!(particleGrid, interpConstant*particleGrid.dx)
    dt = stabilityCondition*getTimeStep(particleGrid, eq, interpAlpha, interpConstant*particleGrid.dx)

    # Get ODE matrices & compute eigenvalues
    A1 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "central", 1, weightFunc))
    lambdas1 = eigvals(A1)

    A2 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "central", 2, weightFunc))
    lambdas2 = eigvals(A2)

    A3 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "upwind", 1, weightFunc))
    lambdas3 = eigvals(A3)

    A4 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "upwind", 2, weightFunc))
    lambdas4 = eigvals(A4)

    A5 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "muscl", 1, weightFunc))
    lambdas5 = eigvals(A5)

    A6 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "muscl", 2, weightFunc))
    lambdas6 = eigvals(A6)

    A7 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "muscl", 3, weightFunc))
    lambdas7 = eigvals(A7)

    A8 = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, "muscl", 4, weightFunc))
    lambdas8 = eigvals(A8)

    if typeof(weightFunc) == exponentialWeightFunction
        wString = "exp weight"
    elseif typeof(weightFunc) == inverseWeightFunction
        wString = "inv weight"
    else
        error("Weightfunction unknown")
    end

    # Plotting of eigenvalues
    ms = 3
    p1 = plot(real.(lambdas1*dt), imag.(lambdas1*dt), seriestype=:scatter, legend=false, ylabel=L"Im(\lambda dt)", xlabel=L"Re(\lambda dt)", label="Central: order 1", ms=ms, aspect_ratio=:equal, framestyle=:origin, markershape=:circle)
    plot!(p1, real.(lambdas2*dt), imag.(lambdas2*dt), label="Central: order 2", seriestype=:scatter, ms=ms, markershape=:circle)
    plot!(p1, real.(lambdas3*dt), imag.(lambdas3*dt), label="Upwind: order 1", seriestype=:scatter, ms=ms, markershape=:cross)
    plot!(p1, real.(lambdas4*dt), imag.(lambdas4*dt), label="Upwind: order 2", seriestype=:scatter, ms=ms, markershape=:cross)
    plot!(p1, real.(lambdas5*dt), imag.(lambdas5*dt), label="MUSCL w. order 1 linear rec.", seriestype=:scatter, ms=ms, markershape=:diamond)
    plot!(p1, real.(lambdas6*dt), imag.(lambdas6*dt), label="MUSCL w. order 2 quadratic rec.", seriestype=:scatter, ms=ms, markershape=:diamond)
    plot!(p1, real.(lambdas7*dt), imag.(lambdas7*dt), label="MUSCL w. order 3 cubic rec.", seriestype=:scatter, ms=ms, markershape=:diamond)
    plot!(p1, real.(lambdas8*dt), imag.(lambdas8*dt), label="MUSCL w. order 4 quartic rec.", seriestype=:scatter, ms=ms, markershape=:diamond)
    
    # Add stability regions
    xminPlot, xmaxPlot = xlims(p1)
    yminPlot, ymaxPlot = ylims(p1)
    xs = range(xminPlot, xmaxPlot, length=200)
    ys = range(yminPlot, ymaxPlot, length=200)
    Z = xs' .+ ys*im
    plot!(p1, xs, ys, RK1(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK1 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    plot!(p1, xs, ys, RK2(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK2 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    plot!(p1, xs, ys, RK3(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK3 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    plot!(p1, xs, ys, RK4(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK4 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    
    savefig(p1, saveDir*"$(saveName).pdf")

    # Zoom 
    ms = 3
    p2 = plot(real.(lambdas1*dt), imag.(lambdas1*dt), seriestype=:scatter, ylabel=L"Im(\lambda dt)", xlabel=L"Re(\lambda dt)", label="Central: order 1", ms=ms, xlims = (-0.005, 0.005), ylims = (-0.25, 0.25), framestyle=:origin, markershape=:circle)
    plot!(p2, real.(lambdas2*dt), imag.(lambdas2*dt), label="Central: order 2", seriestype=:scatter, ms=ms, markershape=:circle)
    plot!(p2, real.(lambdas3*dt), imag.(lambdas3*dt), label="Upwind: order 1", seriestype=:scatter, ms=ms, markershape=:cross)
    plot!(p2, real.(lambdas4*dt), imag.(lambdas4*dt), label="Upwind: order 2", seriestype=:scatter, ms=ms, markershape=:cross)        
    plot!(p2, real.(lambdas5*dt), imag.(lambdas5*dt), label="MUSCL: order 1", seriestype=:scatter, ms=ms, markershape=:diamond)
    plot!(p2, real.(lambdas6*dt), imag.(lambdas6*dt), label="MUSCL: order 2", seriestype=:scatter, ms=ms, markershape=:diamond)     
    plot!(p2, real.(lambdas7*dt), imag.(lambdas7*dt), label="MUSCL: order 3", seriestype=:scatter, ms=ms, markershape=:diamond)
    plot!(p2, real.(lambdas8*dt), imag.(lambdas8*dt), label="MUSCL: order 4", seriestype=:scatter, ms=ms, markershape=:diamond)
    
    # Adding stability regions
    xminPlot, xmaxPlot = xlims(p2)
    yminPlot, ymaxPlot = ylims(p2)
    xs = range(xminPlot, xmaxPlot, length=200)
    ys = range(yminPlot, ymaxPlot, length=200)
    Z = xs' .+ ys*im
    plot!(p2, xs, ys, RK1(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK1 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    plot!(p2, xs, ys, RK2(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK2 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    plot!(p2, xs, ys, RK3(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK3 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    plot!(p2, xs, ys, RK4(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK4 stability region", lw=2, color=:turbo, clims=(-2, 1.1))
    
    savefig(p2, saveDir*"$(saveName)Zoom.pdf")

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
end


function doSimluation(initFunc::String, AlgID::Integer, seed::Integer, weightFunc = exponentialWeightFunction())
    saveDir = "$(@__DIR__)/data/"

    # Equation
    eq = LinearAdvection(a)

    # Grid stuff
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = (xmax - xmin)/(2*N), rng=MersenneTwister(seed))
    dtEuler = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*particleGrid.dx)
    CFL = 0.05

    # Simlation algorithms
    if AlgID == 1
        method = EulerUpwind(N; weightFunction=weightFunc)
        saveDir *= "MeshfreeUpwind1"
    elseif AlgID == 2
        method = RK3(CentralGradient(1; weightFunction=weightFunc), N)
        saveDir *= "RK3Central1"
    elseif AlgID == 3
        method = RK3(CentralGradient(1; weightFunction=weightFunc), N; mood = MOODu1(deltaRelax=true))
        saveDir *= "RK3MOODCentral1"
    elseif AlgID == 4
        method = RK3(UpwindGradient(1; weightFunction=weightFunc), N)
        saveDir *= "RK3Upwind1"
    elseif AlgID == 5
        method = RK3(CentralGradient(2; weightFunction=weightFunc), N)
        saveDir *= "RK3Central2"
    elseif AlgID == 6
        method = RK3(CentralGradient(2; weightFunction=weightFunc), N; mood = MOODu1(deltaRelax=true))
        saveDir *= "RK3MOODCentral2"
    elseif AlgID == 7
        method = RK3(UpwindGradient(2; weightFunction=weightFunc), N)
        saveDir *= "RK3Upwind2"
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
        method = RalstonRK2(MUSCL(1; weightFunction=weightFunc), N)
        saveDir *= "RK2MUSCL1"  # MUSCL with linear reconstruction
    elseif AlgID == 12
        method = RalstonRK2(MUSCL(2; weightFunction=weightFunc), N)  # MUSCL with quadratic reconstruction
        saveDir *= "RK2MUSCL2"
    elseif AlgID == 13
        method = RK3(MUSCL(3; weightFunction=weightFunc), N)  # MUSCL with cubic reconstruction
        saveDir *= "RK3MUSCL3"
    elseif AlgID == 14
        method = RK4(MUSCL(4; weightFunction=weightFunc), N)  # MUSCL with quartic reconstruction
        saveDir *= "RK4MUSCL4"
    elseif AlgID == 15
        method = RalstonRK2(MUSCL(2; weightFunction=weightFunc), N; mood = MOODu1(deltaRelax=true))  # MUSCL with quadratic reconstruction
        saveDir *= "RK2MUSCL2MOOD"
    end
    saveDir *= "_irgrid_$(seed)"

    # Initial condition
    if initFunc == "smoothInit"
        setInitialConditions!(particleGrid, smoothInit)
        saveDir *= "_smoothInit"
    elseif initFunc == "shockInit"
        setInitialConditions!(particleGrid, shockInit)
        saveDir *= "_shockInit"
    end

    # Create SimSettings object
    saveFreq = 200
    settings = SimSetting(  tmax=tmax,
                            dt=dtEuler*CFL,
                            interpRange=interpRangeConst*particleGrid.dx,
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq)

    # Do simulation
    failed = false
    try
        mainTimeIntegrator!(method, eq, particleGrid, settings)
    catch e
        println("Error caught: $(e) in alg $(AlgID), seed: $(seed)")
        failed = true
    end

    plotDensity(settings, settings.currentSaveNb-1; saveFigure=true)
    animateDensity(settings; saveFigure=true, fps=2)

    # Only keep final grid
    cleanDir(settings.saveDir)

    if failed 
        return false
    else
        rhos = map(particle -> particle.rho, particleGrid.grid)
        if any(isnan, rhos) || any(isinf, rhos)
            return false
        end
        if any(abs.(rhos) .> 4)
            return false
        end
        return true
    end
end

end  # everywhere

function runSimulations()
    algs = [7; 8; 12; 14; 15]
    labels = ["Upwind 2"; "WENO 2"; "MUSCL 2"; "MUSCL 4"; "MUSCL 2 MOOD"]
    repeats = 100
    simArgs = [(seed, alg) for alg in algs for seed in 1:repeats]

    # # Do simulations
    # result = map(eachindex(simArgs)) do i
    #     doSimluation("shockInit", simArgs[i][2], simArgs[i][1])
    # end
    # save("$(@__DIR__)/data/simulationStability.jld2", "results", reshape(result, (repeats, length(algs))), "labels", labels)

    # # Compute spectra
    # labels = ["Upwind 2"; "MUSCL 2"; "MUSCL 4"]
    # resultSpectra = zeros(Bool, (repeats, 3))
    # result = map(eachindex(1:repeats)) do i
    #     temp = checkStability("stabilityTest", interpRangeConst, exponentialWeightFunction(), i) 
    #     resultSpectra[i, :] .= temp
    # end
    # save("$(@__DIR__)/data/eigenvalueStability.jld2", "results", resultSpectra, "labels", labels)

    computeStabilities("eigenvalues1", interpRangeConst, exponentialWeightFunction(), true)
    computeStabilities("eigenvalues2", interpRangeConst, inverseWeightFunction(), true)
    computeStabilities("eigenvalues3", interpRangeConst, exponentialWeightFunction(), false)
end

runSimulations()
