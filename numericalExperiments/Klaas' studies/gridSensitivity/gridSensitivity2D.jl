using Distributed

@everywhere begin
    using Plots
    using LinearAlgebra
    using SparseArrays
    using Arpack
    using Random
    using JLD2
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
global const xmin = -5;
global const xmax = 5;
global const a = (1.0, 1.0);

function computeStability(interpConstant::Real, N::Integer, weightFunction::MLSWeightFunction, seed::Integer, noiseLevel::Integer, interpAlpha::Float64)
    tol = 1e-13

    # Generate grid
    dx = (xmax-xmin)/N
    rng = MersenneTwister(seed)
    if noiseLevel == 1
        randomness = dx/2
    elseif noiseLevel == 2
        randomness = 0.25*dx
    elseif noiseLevel == 3
        randomness = 0.15*dx
    elseif noiseLevel == 4
        randomness = 0.1*dx
    elseif noiseLevel == 5
        randomness = 0.05*dx
    elseif noiseLevel == 6
        randomness = 0.005*dx
    elseif noiseLevel == 7
        randomness = 0.0005*dx
    elseif noiseLevel == 8
        randomness = 0.0*dx
    end
    particleGrid = ParticleGrid2D(xmin, xmax, xmin, xmax, N, N; randomness = (randomness, randomness), rng=rng)  
    updateNeighbours!(particleGrid, interpConstant*particleGrid.dx)

    # For multiple algorithms, check if there are unstable eigenvalues
    algs = ["upwindPraveen"; "upwindClassic"; "upwindClassic"; "muscl"; "muscl"]
    orders = [1; 1; 2; 1; 2]
    r1 = zeros(Int64, length(algs))  # Amount of unstable eigenvalues per algorithm
    r2 = zeros(Complex, length(algs))  # If at least one unstable complex eigenvalue, return the one with the largest real part
    for (i, (alg, order)) in enumerate(zip(algs, orders))
        A = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, alg, order, weightFunction))

        # Exact solver
        lambdas = eigvals(A)
        lambdaReal = real.(lambdas)
        if any(lambdaReal .>= tol)
            r1[i] = count(lambdaReal .>= tol)
            r2[i] = maximum(real, lambdas)
        end
    end
    return r1, r2, particleGrid
end

end  # @everywhere

function checkGridSensitivity()
    @assert length(ARGS) >= 2 "Give CLI arguments: amount of runs, Ni, Nj, ..."
    interpConst = sqrt(5.0^2 + 3.0^2)
    runs = parse(Int64, ARGS[1])
    
    # Ns
    Ns = [parse(Int64, ARGS[i]) for i = 2:length(ARGS)]

    # Weight function logic
    weightFunctions = [exponentialWeightFunction()]
    weightStr = ["expWeight"]

    # Noise logic
    noise = range(1, 8)
    noiseNames = ["Noise$(x)" for x in noise]

    # Alpha logic
    alphaStrings = ["alpha1"]
    alphas = [6.0]

    println("Runs: $(runs)")
    println("Ns: $(Ns)")

    j = 1
    for (alpha, alphaString) in zip(alphas, alphaStrings)
        for (noiseLevel, noiseName) in zip(noise, noiseNames)
            for (weightFunction, wStr) in zip(weightFunctions, weightStr)
                for N in Ns
                    res = Matrix{Int64}(undef, runs, 5)
                    eigs = Matrix{Complex}(undef, runs, 5)

                    pmapResult = pmap(eachindex(1:runs)) do run
                        r1, r2, _ = computeStability(interpConst, N, weightFunction, run, noiseLevel, alpha)
                        return (r1, r2)
                    end

                    for (i, (r1, r2)) in enumerate(pmapResult)
                        res[i, :] .= r1
                        eigs[i, :] .= r2
                    end

                    println("$(j*100/(length(alphaStrings)*length(noiseNames)*length(Ns)*length(weightFunctions)))% done.")
                    j += 1

                    # for run = 1:runs
                    #     r1, r2, _ = computeStability(interpConst, N, weightFunction)
                    #     res[run, :] .= r1
                    #     eigs[run, :] .= r2
                    #     if mod(run, 10) == 0
                    #         println("$(100*j/(runs*length(Ns)*length(weightFunctions)))% done.")
                    #     end
                    #     j += 1
                    # end
                    save("$(@__DIR__)/data2D$(noiseName)/gridSensitivity2D_$(N)_$(wStr)_$(alphaString).jld2", "res", res, "eigs", eigs, "N", N, "interpConst", interpConst, "alphas", alphas, "alphaStrings", alphaStrings)
                end
            end
        end
    end
end

checkGridSensitivity()