using Distributed
using ProgressMeter

@everywhere begin
    using JLD2
    using FileIO
    using LinearAlgebra
    using SparseArrays
    using LaTeXStrings
    using Random
    using Plots
    using Meshfree4ScalarEq.ScalarHyperbolicEquations
    using Meshfree4ScalarEq.ParticleGrids
    using Meshfree4ScalarEq.TimeIntegration
    using Meshfree4ScalarEq.Interpolations
    using Meshfree4ScalarEq.SimSettings
    using Meshfree4ScalarEq.ParticleGridStability
    using Meshfree4ScalarEq
end

@everywhere begin

const global labels = ["Upwind Classic: order 2" "MUSCL 2" "MUSCL 1" "Upwind Classic: order 1"]
const global algorithms = ["upwindClassic"; "muscl"; "muscl"; "upwindClassic"]
const global orders = [2; 2; 1; 1]

function computeStabilities(saveName::String, stabilityCondition::Float64, N::Int64, interpRangeConst::Float64, weightFunction::MLSWeightFunction, seed::Integer)
    saveDir = "$(@__DIR__)/data/"
    xmin = -5
    xmax = 5
    interpAlpha = 6.0

    if typeof(weightFunction) == exponentialWeightFunction
        saveName *= "_expWeight"
    elseif typeof(weightFunction) == inverseWeightFunction
        saveName *= "_inverseWeight"
    end
    textPath = saveDir*saveName*".txt"

    if isfile(textPath)
        rm(textPath)
    end

    # Stability functions of RK1 (euler), RK2 and RK3
    RK1(z) = @. abs(1.0 + z)
    RK2(z) = @. abs(1.0 + z + 0.5*z^2) 
    RK3(z) = @. abs(1.0 + z + 0.5*z^2 + (z^3)/6) 

    rng = MersenneTwister(seed) 

    # Generate grid
    a = (1.0, 1.0)
    eq = LinearAdvection(a)

    dx = (xmax-xmin)/N
    dy = (xmax-xmin)/N
    particleGrid = ParticleGrid2D(xmin, xmax, xmin, xmax, N, N; randomness = (dx/2, dy/2), rng=rng)  
    dt = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*dx)*stabilityCondition    
    updateNeighbours!(particleGrid, interpRangeConst*particleGrid.dx)

    lambdas = Vector{Vector{Complex}}(undef, length(algorithms))
    stability = zeros(length(labels))

    for i in eachindex(algorithms)
        A1 = collect(computeODE(particleGrid, a, interpAlpha, interpRangeConst*dx, algorithms[i], orders[i], weightFunction))
        lambdas[i] = eigvals(A1)
        stability[i] = all(real.(lambdas[i]) .< 1e-13)
        if any(real.(lambdas[i]) .> 1e-13)
            t = "Unstable eigenvalues detected in $(algorithms[i]), order $(orders[i]) for N=$(N): $(lambdas[i][real.(lambdas[i]) .> 0.0])\n"
            println(t)
            open(textPath, "a") do file
                write(file, t)
            end
        end
    end

    # Plotting of eigenvalues
    ms = 3
    p1 = plot(real.(lambdas[1]*dt), imag.(lambdas[1]*dt), seriestype=:scatter, ylabel=L"Im(\lambda dt)", xlabel=L"Re(\lambda dt)", label=labels[1], ms=ms, aspect_ratio=:equal, title="Eigenvalues for N = $(N), h=$(interpRangeConst)dx, CFL=$(stabilityCondition)", framestyle=:origin)
    for i = 2:length(orders)
        plot!(p1, real.(lambdas[i]*dt), imag.(lambdas[i]*dt), label=labels[i], seriestype=:scatter, ms=ms)
    end
    
    # Add stability regions
    xminPlot, xmaxPlot = xlims(p1)
    yminPlot, ymaxPlot = ylims(p1)
    xs = range(xminPlot, xmaxPlot, length=200)
    ys = range(yminPlot, ymaxPlot, length=200)
    Z = xs' .+ ys*im
    plot!(p1, xs, ys, RK1(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK1 stability region", lw=2)
    plot!(p1, xs, ys, RK2(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK2 stability region", lw=2)
    plot!(p1, xs, ys, RK3(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK3 stability region", lw=2)
    savefig(p1, saveDir*"$(saveName).pdf")

    return stability
end

function computeStabilitiesPaper(saveName::String, stabilityCondition::Float64, N::Int64, interpRangeConst::Float64, weightFunction::MLSWeightFunction, randomnessFac::Real, seed::Int64)
    saveDir = "$(@__DIR__)/"
    xmin = -5
    xmax = 5
    interpAlpha = 6

    if typeof(weightFunction) == exponentialWeightFunction
        saveName *= "_expWeight"
    elseif typeof(weightFunction) == inverseWeightFunction
        saveName *= "_inverseWeight"
    end
    textPath = saveDir*saveName*".txt"

    if isfile(textPath)
        rm(textPath)
    end

    rng = MersenneTwister(seed)

    # Stability functions of RK1 (euler), RK2 and RK3
    RK1(z) = @. abs(1.0 + z)
    RK2(z) = @. abs(1.0 + z + 0.5*z^2) 
    RK3(z) = @. abs(1.0 + z + 0.5*z^2 + (z^3)/6) 

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Generate grid
    a = (1.0, 1.0)
    eq = LinearAdvection(a)

    dx = (xmax-xmin)/N
    dy = (xmax-xmin)/N
    particleGrid = ParticleGrid2D(xmin, xmax, xmin, xmax, N, N; randomness = (dx*randomnessFac, dy*randomnessFac), rng=rng)  
    dt = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*dx)*stabilityCondition    
    updateNeighbours!(particleGrid, interpRangeConst*particleGrid.dx)

    labels = ["Upwind: order 2" "MUSCL: order 2" "MUSCL: order 1" "Upwind: order 1"]
    algorithms = ["upwindClassic"; "muscl"; "muscl"; "upwindPraveen"]
    orders = [2; 2; 1; 1]
    lambdas = Vector{Vector{Complex}}(undef, length(algorithms))

    for i in eachindex(algorithms)
        A1 = collect(computeODE(particleGrid, a, interpAlpha, interpRangeConst*dx, algorithms[i], orders[i], weightFunction))
        lambdas[i] = eigvals(A1)
        if any(real.(lambdas[i]) .> 1e-13)
            t = "Unstable eigenvalues detected in $(algorithms[i]), order $(orders[i]) for N=$(N): $(lambdas[i][real.(lambdas[i]) .> 0.0])\n"
            println(t)
            open(textPath, "a") do file
                write(file, t)
            end
        end
    end

    # Plotting of eigenvalues
    ms = 2
    p1 = plot(real.(lambdas[1]*dt), imag.(lambdas[1]*dt), seriestype=:scatter, ylabel=L"Im(\lambda dt)", xlabel=L"Re(\lambda dt)", label=labels[1], ms=ms, aspect_ratio=:equal, framestyle=:origin, legend=:bottomright)
    for i = 2:length(orders)
        plot!(p1, real.(lambdas[i]*dt), imag.(lambdas[i]*dt), label=labels[i], seriestype=:scatter, ms=ms)
    end
    
    # Add stability regions
    xminPlot = -2.0
    xmaxPlot = 0.4
    yminPlot = -1.4
    ymaxPlot = 1.4
    xs = range(xminPlot, xmaxPlot, length=200)
    ys = range(yminPlot, ymaxPlot, length=200)
    Z = xs' .+ ys*im
    plot!(p1, xs, ys, RK1(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK1 stability region", lw=2)
    plot!(p1, xs, ys, RK2(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK2 stability region", lw=2)
    plot!(p1, xs, ys, RK3(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK3 stability region", lw=2)
    
    savefig(p1, saveDir*"$(saveName).pdf")

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
end

end  # @everywhere

function plotStabilityRegions()
    interpRange = sqrt(5.0^2 + 3.0^2)
    CFL = 0.25

    repeats = 100
    Ns = [70 for _ in 1:repeats]

    simArgs = [("eigenvalues_$(N)_$(Ncount)", N, exponentialWeightFunction(), Ncount) for (Ncount, N) in enumerate(Ns)]
    results = Array{Bool}(undef, (repeats, length(labels)))

    resultsPmap = @showprogress pmap(eachindex(simArgs)) do i
        computeStabilities(simArgs[i][1], CFL, simArgs[i][2], interpRange, simArgs[i][3], simArgs[i][4])
    end

    # Copy results to matrix
    for i in eachindex(simArgs)
        results[i, :] = resultsPmap[i]
    end
    save("$(@__DIR__)/data/eigenvalueStability.jld2", "results", results, "labels", labels)
    
    # computeStabilitiesPaper("eigenvaluesPaper_80", 1/2, 70, interpRange, exponentialWeightFunction(), 0.5, 10)
    # computeStabilitiesPaper("eigenvaluesPaper_80Uniform", 1/2, 70, interpRange, exponentialWeightFunction(), 0, 10)
end

computeStabilitiesPaper("eigenvaluesPaper_80", 1/2, 70, sqrt(5.0^2 + 3.0^2), exponentialWeightFunction(), 0.5, 10)
computeStabilitiesPaper("eigenvaluesPaper_80Uniform", 1/2, 70, sqrt(5.0^2 + 3.0^2), exponentialWeightFunction(), 0, 10)

# plotStabilityRegions()