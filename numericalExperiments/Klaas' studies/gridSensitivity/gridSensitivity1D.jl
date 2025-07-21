using Plots
using LinearAlgebra
using SparseArrays
using Arpack
using JLD2
using Random
using LaTeXStrings
using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq.ParticleGridStability
using Meshfree4ScalarEq

global const xmin = -5;
global const xmax = 5;
global const a = 1.0;
global const interpAlpha = 1.0;

function computeStability(interpConstant::Real, N::Integer, weightFunction::MLSWeightFunction, seed::Integer, randomness::Real)
    # Generate grid
    dx = (xmax-xmin)/N
    rng = MersenneTwister(seed)
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = randomness*dx, rng=rng)
    updateNeighbours!(particleGrid, interpConstant*particleGrid.dx)

    # For multiple algorithms, check if there are unstable eigenvalues
    algs = ["central"; "central"; "upwind"; "upwind"; "muscl"; "muscl"; "muscl"; "muscl"]
    orders = [1; 2; 1; 2; 1; 2; 3; 4]
    r1 = zeros(Int64, length(algs))  # Amount of unstable eigenvalues per algorithm
    r2 = zeros(Complex, length(algs))  # If at least one unstable complex eigenvalue, return the one with the largest real part
    for (i, (alg, order)) in enumerate(zip(algs, orders))
        A = collect(computeODE(particleGrid, a, interpAlpha, interpConstant*particleGrid.dx, alg, order, weightFunction))
        lambdas = eigvals(A)
        lambdaReal = real.(lambdas)
        if any(lambdaReal .>= 1e-13)
            r1[i] = count(lambdaReal .>= 1e-13)
            r2[i] = maximum(real, lambdas)
        end
    end
    return r1, r2, particleGrid
end

function checkGridSensitivity()
    @assert length(ARGS) >= 2 "Give CLI arguments: amount of runs, Ni, Nj, ..."
    interpConst = 3.5
    runs = parse(Int64, ARGS[1])
    
    Ns = [parse(Int64, ARGS[i]) for i = 2:length(ARGS)]
    weightFunctions = [exponentialWeightFunction(); inverseWeightFunction()]
    weightStr = ["expWeight"; "invWeight"]
    randomnessArr = [0.5; 0.5-0.05; 0.5-0.1]  # dx factors e.g. 0.5*dx, 0.5dx-0.05dx, ...
    randomnessName = ["MaxNoise"; "Noise1"; "Noise2"]

    println("Runs: $(runs)")
    println("Ns: $(Ns)")

    j = 1
    for (randomnessIndex, randomness) in enumerate(randomnessArr)        
        for (weightFunction, wStr) in zip(weightFunctions, weightStr)
            for N in Ns
                res = Matrix{Int64}(undef, runs, 8)
                eigs = Matrix{Complex}(undef, runs, 8)
                for run = 1:runs
                    r1, r2, _ = computeStability(interpConst, N, weightFunction, run, randomness)
                    res[run, :] .= r1
                    eigs[run, :] .= r2
                    if mod(run, 10) == 0
                        println("$(100*j/(runs*length(Ns)*length(weightFunctions)*length(randomnessArr)))% done.")
                    end
                    j += 1
                end
                save("$(@__DIR__)/data1D$(randomnessName[randomnessIndex])/gridSensitivity1D_$(N)_$(wStr).jld2", "res", res, "eigs", eigs, "N", N, "interpConst", interpConst)
            end
        end
    end
end

checkGridSensitivity()