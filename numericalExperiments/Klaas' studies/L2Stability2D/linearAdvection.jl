using Distributed
@everywhere begin
    using ProgressMeter
    using JLD2
    using FileIO
    using Random
    using LaTeXStrings
    using Meshfree4ScalarEq.ScalarHyperbolicEquations
    using Meshfree4ScalarEq.ParticleGrids
    using Meshfree4ScalarEq.TimeIntegration
    using Meshfree4ScalarEq.Interpolations
    using Meshfree4ScalarEq.SimSettings
    using Meshfree4ScalarEq
end

@everywhere begin
function smoothInit(x::Real, y::Real)
    return exp(-x^2 - y^2)
end

function shockInit(x::Real, y::Real)
    return (-0.5 < x < 0.5) && (-0.5 < y < 0.5) ? 1.0 : 0.0
end


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
    
function doSimulation(initFunc::String, AlgID::Integer, N::Integer, weightFunction::MLSWeightFunction, seed::Integer, interpRangeConst::Real)
    # Simulation settings
    tmax = 30*sqrt(2)  # One loop along diagonal
    xmin = -5
    xmax = 5
    saveDir = "$(@__DIR__)/data/"
    interpAlpha = 6.0
    saveFreq = 50

    rng = MersenneTwister(seed)

    # Simlation algorithms
    CFL = 0.1
    if AlgID == 1
        method = EulerUpwind(N, N; algType="Classic", weightFunction=weightFunction)
        saveDir *= "MeshfreeUpwindClassic"
    elseif AlgID == 2
        method = EulerUpwind(N, N; algType="Tiwari", weightFunction=weightFunction)
        saveDir *= "MeshfreeUpwindTiwari"
    elseif AlgID == 3
        method = EulerUpwind(N, N; algType="Praveen", weightFunction=weightFunction)
        saveDir *= "PraveenUpwind"
    elseif AlgID == 4
        method = RK3(WENO(2; weightFunction=weightFunction), N, N) 
        saveDir *= "RK3WENO"
    elseif AlgID == 5
        method = RK3(UpwindGradient(2; algType="Classic", weightFunction=weightFunction), N, N)  
        saveDir *= "RK3ClassicUpwind2"
    elseif AlgID == 6
        method = RK3(UpwindGradient(1; algType="Classic", weightFunction=weightFunction), N, N) 
        saveDir *= "RK3ClassicUpwind1"
    elseif AlgID == 7
        method = RK3(MUSCL(1; weightFunction=weightFunction), N, N) 
        saveDir *= "MUSCL1"
    elseif AlgID == 8
        method = RK3(MUSCL(2; weightFunction=weightFunction), N, N) 
        saveDir *= "MUSCL2"
    elseif AlgID == 9
        method = RK3(MUSCL(2; weightFunction=weightFunction), N, N, mood=MOODu2(deltaRelax=true)) 
        saveDir *= "MUSCL2MOOD"
    elseif AlgID == 10
        method = RK3(DumbserWENO(; weightFunction=weightFunction), N, N)
        saveDir *= "DumbserWENO"
    end

    # Equation
    a = (1.0, 1.0)
    eq = LinearAdvection(a)
    
    # Grid stuff
    dx = (xmax-xmin)/N
    dy = (xmax-xmin)/N
    particleGrid = ParticleGrid2D(xmin, xmax, xmin, xmax, N, N; randomness = (dx/2, dy/2), rng=rng)  
    dt = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*dx)*CFL    
    saveDir *= "_$(N)"

    # Initial condition
    if initFunc == "smoothInit"
        setInitialConditions!(particleGrid, smoothInit)
        saveDir *= "_smoothInit"
    elseif initFunc == "shockInit"
        setInitialConditions!(particleGrid, shockInit)
        saveDir *= "_shockInit"
    end
    saveDir *= "_$(seed)"

    # Create SimSettings object
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRangeConst*dx,
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq)

    # Do simulation
    failed = false
    try
        mainTimeIntegrator!(method, eq, particleGrid, settings)
    catch e
        println("Error caught: $(e) in algorithm $(AlgID) with N = $(N), seed = $(seed)")
        println("Skipping the rest of the simulation.")
        failed = true
    end

    # Generate plots
    try
        ms = 4
        plotDensity(settings, 0; saveFigure=true, ms=ms)
        plotDensity(settings, settings.currentSaveNb-1; saveFigure=true, ms=ms)
        animateDensity(settings; saveFigure=true, fps=2, ms=ms)
    catch e
        println("Plotting error $(e) in algorithm $(AlgID) with N = $(N), seed = $(seed)")
    end

    # Only keep final grid
    cleanDir(settings.saveDir)

    if failed
        return false
    else
        rhos = map(particle -> particle.rho, particleGrid.grid)
        if any(isnan, rhos) || any(isinf, rhos)
            return false
        end
        if any(abs.(rhos) .> 10)
            return false
        end
        return true
    end
end

end  # @everywhere

function doDistributedSimulations()
    interpConst = sqrt(5.0^2 + 3.0^2)
    repeats = 100
    labels = ["SS WENO 2"; "SS Upwind Classic 2"; "SS MUSCL 1"; "SS MUSCL 2"; "SS MUSCL 2 MOOD"; "SS DumbserWENO 2"]
    simAlgs = [4; 5; 7; 8; 9; 10]

    Ns = [70 for _ in 1:repeats]

    simArgs = [("shockInit", j, N, Ncount) for j in simAlgs for (Ncount, N) in enumerate(Ns)]
    result = @showprogress pmap(eachindex(simArgs)) do i
        doSimulation(simArgs[i][1], simArgs[i][2], simArgs[i][3], exponentialWeightFunction(), simArgs[i][4], interpConst)
    end

    save("$(@__DIR__)/data/simulationStability.jld2", "results", reshape(result, (length(Ns), length(simAlgs))), "labels", labels)
end

doDistributedSimulations()
