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
global const interpAlpha = 6.0;
global const algs = ["central"; "central"; "upwind"; "upwind"; "muscl"; "muscl"; "muscl"; "muscl"]
global const orders = [1; 2; 1; 2; 1; 2; 3; 4]

function computeStabilities(saveName::String, interpConstant::Real, weightFunc::MLSWeightFunction, N::Integer, randomness::Real) 
    saveDir = "$(@__DIR__)/data1D/"
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
    dx = (xmax-xmin)/N
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness=randomness*dx, singlePointPerturbation=true)
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
    ms = 4
    p1 = plot(real.(lambdas3*dt), imag.(lambdas3*dt), seriestype=:scatter, legend=false, ylabel=L"Im(\lambda dt)", xlabel=L"Re(\lambda dt)", label="Upwind: order 1", ms=ms, aspect_ratio=:equal, framestyle=:origin, markershape=:cross)
    plot!(p1, real.(lambdas4*dt), imag.(lambdas4*dt), label="Upwind: order 2", seriestype=:scatter, ms=ms, markershape=:cross)
    # plot!(p1, real.(lambdas1*dt), imag.(lambdas1*dt), label="Central: order 1", seriestype=:scatter, ms=ms, markershape=:circle)
    # plot!(p1, real.(lambdas2*dt), imag.(lambdas2*dt), label="Central: order 2", seriestype=:scatter, ms=ms, markershape=:circle)
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
    ms = 4
    p2 = plot(real.(lambdas3*dt), imag.(lambdas3*dt), seriestype=:scatter, ylabel=L"Im(\lambda dt)", xlabel=L"Re(\lambda dt)", label="Upwind: order 1", ms=ms, xlims = (-0.00125, 0.00125), ylims = (-0.25, 0.25), framestyle=:origin, markershape=:cross)
    plot!(p2, real.(lambdas4*dt), imag.(lambdas4*dt), label="Upwind: order 2", seriestype=:scatter, ms=ms, markershape=:cross)        
    # plot!(p2, real.(lambdas2*dt), imag.(lambdas2*dt), label="Central: order 2", seriestype=:scatter, ms=ms, markershape=:circle)
    # plot!(p2, real.(lambdas1*dt), imag.(lambdas1*dt), label="Central: order 1", seriestype=:scatter, ms=ms, markershape=:circle)
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


function computeStability(interpConstant::Real, N::Integer, weightFunction::MLSWeightFunction, seed::Integer, randomness::Real)
    # Generate grid
    dx = (xmax-xmin)/N
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = randomness*dx, singlePointPerturbation=true)
    updateNeighbours!(particleGrid, interpConstant*particleGrid.dx)

    # For multiple algorithms, check if there are unstable eigenvalues
    r1 = zeros(Int64, length(algs))  # Amount of unstable eigenvalues per algorithm
    r2 = zeros(Complex, length(algs))  # If at least one unstable complex eigenvalue, return the one with the largest real part
    for (i, (alg, order)) in enumerate(zip(algs, orders))
        A = collect(computeODE(particleGrid, a, interpAlpha, alg, order, weightFunction))
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
    @assert length(ARGS) >= 2 "Give CLI arguments: grid sizes Ni, Nj, ..."
    interpConst = 3.0
    
    Ns = [parse(Int64, ARGS[i]) for i = 1:length(ARGS)]
    weightFunctions = [exponentialWeightFunction()]
    weightStr = ["expWeight"]
    randomnessArr = range(-0.499, 0.499, 100)

    res = Array{Float64}(undef, (length(randomnessArr), length(weightFunctions), length(Ns), 8))
    eigs = Array{Float64}(undef, (length(randomnessArr), length(weightFunctions), length(Ns), 8))

    println("Ns: $(Ns)")

    j = 1
    for (randomnessIndex, randomness) in enumerate(randomnessArr)        
        for (weightIndex, (weightFunction, wStr)) in enumerate(zip(weightFunctions, weightStr))
            for (NIndex, N) in enumerate(Ns)
                r1, r2, _ = computeStability(interpConst, N, weightFunction, j, randomness)
                res[randomnessIndex, weightIndex, NIndex, :] .= r1
                eigs[randomnessIndex, weightIndex, NIndex, :] .= r2
                if mod(j, 30) == 0
                    println("$(100*j/(length(Ns)*length(weightFunctions)*length(randomnessArr)))% done.")
                end
                j += 1
            end
        end
    end
    save("$(@__DIR__)/data1D/gridSensitivity1D_singlePerturbation.jld2", "res", res, "eigs", eigs, "Ns", Ns, "interpConst", interpConst, "weightStr", weightStr, "randomnessArr", randomnessArr, "algs", algs, "orders", orders)
end

# checkGridSensitivity()
computeStabilities("spectra0", 2.1, exponentialWeightFunction(), 100, 0.05) 
computeStabilities("spectra1", 3.0, exponentialWeightFunction(), 100, 0.05) 
computeStabilities("spectra2", 3.1, exponentialWeightFunction(), 100, 0.05) 
computeStabilities("spectra3", 4.0, exponentialWeightFunction(), 100, 0.05) 
computeStabilities("spectra4", 4.1, exponentialWeightFunction(), 100, 0.05) 
