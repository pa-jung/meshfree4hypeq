using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq.ParticleGridStability
using Meshfree4ScalarEq
using Plots
using LinearAlgebra
using SparseArrays
using Arpack
using LaTeXStrings

const global N = 100;
const global xmin = -5;
const global xmax = 5;
const global a = 1.0;
const global interpAlpha = 6.0;
const global interpRangeConst = 3.5;
const global tmax = 200.0;

function computeStabilities(saveName::String, interpConstant::Real, weightFunc::MLSWeightFunction) 
    saveDir = "$(@__DIR__)/figures/"
    stabilityCondition = 1.0

    # Stability regions of RK2 and RK3
    RK1(z) = @. abs(1.0 + z) 
    RK2(z) = @. abs(1.0 + z + 0.5*z^2) 
    RK3(z) = @. abs(1.0 + z + 0.5*z^2 + (z^3)/6) 

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Generate grid
    eq = LinearAdvection(a)
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = (xmax - xmin)/(2*N))
    updateNeighbours!(particleGrid, interpConstant*particleGrid.dx)
    dt = stabilityCondition*getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*particleGrid.dx)

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

    if typeof(weightFunc) == exponentialWeightFunction
        wString = "exp weight"
    elseif typeof(weightFunc) == inverseWeightFunction
        wString = "inv weight"
    else
        error("Weightfunction unknown")
    end

    # Plotting of eigenvalues
    ms = 4
    p1 = plot(real.(lambdas1*dt), imag.(lambdas1*dt), seriestype=:scatter, ylabel=L"Im(\lambda dt)", xlabel=L"Re(\lambda dt)", label="Central: order 1", ms=ms, aspect_ratio=:equal, framestyle=:origin)
    plot!(p1, real.(lambdas2*dt), imag.(lambdas2*dt), label="Central: order 2", seriestype=:scatter, ms=ms)
    plot!(p1, real.(lambdas3*dt), imag.(lambdas3*dt), label="Upwind: order 1", seriestype=:scatter, ms=ms)
    plot!(p1, real.(lambdas4*dt), imag.(lambdas4*dt), label="Upwind: order 2", seriestype=:scatter, ms=ms)
    # plot!(p1, real.(lambdas5*dt), imag.(lambdas5*dt), label="MUSCL w. order 1 linear rec.", seriestype=:scatter, ms=ms)
    # plot!(p1, real.(lambdas6*dt), imag.(lambdas6*dt), label="MUSCL w. order 2 linear rec.", seriestype=:scatter, ms=ms)
    
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

    # Zoom 
    ms = 2
    p2 = plot(real.(lambdas1*dt), imag.(lambdas1*dt), seriestype=:scatter, ylabel=L"Im(\lambda dt)", xlabel=L"Re(\lambda dt)", label="Central: order 1", ms=ms, xlims = (-0.05, 0.05), framestyle=:origin)
    plot!(p2, real.(lambdas2*dt), imag.(lambdas2*dt), label="Central: order 2", seriestype=:scatter, ms=ms)
    plot!(p2, real.(lambdas3*dt), imag.(lambdas3*dt), label="Upwind: order 1", seriestype=:scatter, ms=ms)
    plot!(p2, real.(lambdas4*dt), imag.(lambdas4*dt), label="Upwind: order 2", seriestype=:scatter, ms=ms)        
    # plot!(p2, real.(lambdas5*dt), imag.(lambdas5*dt), label="MUSCL w. order 1 linear rec.", seriestype=:scatter, ms=ms)
    # plot!(p2, real.(lambdas6*dt), imag.(lambdas6*dt), label="MUSCL w. order 2 linear rec.", seriestype=:scatter, ms=ms)     
    
    # Adding stability regions
    xminPlot, xmaxPlot = xlims(p2)
    yminPlot, ymaxPlot = ylims(p2)
    xs = range(xminPlot, xmaxPlot, length=200)
    ys = range(yminPlot, ymaxPlot, length=200)
    Z = xs' .+ ys*im
    plot!(p2, xs, ys, RK1(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK1 stability region", lw=2)
    plot!(p2, xs, ys, RK2(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK2 stability region", lw=2)
    plot!(p2, xs, ys, RK3(Z), seriestype=:contour, levels=[1.0;], cbar=false, label="RK3 stability region", lw=2)
    
    savefig(p2, saveDir*"$(saveName)Zoom.pdf")

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
end

computeStabilities("stability", 3.5, exponentialWeightFunction())