using Plots
using Statistics
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
    return sin(x*pi/5)*cos(2*y*pi/5)
    # return sin(y*pi/5)
    # return sin(x*pi/5)
end

function gradSmoothInit(x::Real, y::Real)
    return (pi*cos(pi*x/5)*cos(2*y*pi/5)/5, -2*pi*sin(x*pi/5)*sin(2*y*pi/5)/5)
    # return (0.0, cos(y*pi/5)*pi/5)
    # return (cos(x*pi/5)*pi/5, 0.0)
end

function computeError(j, N)
    a = (1.0, 1.0);
    tmax = 7.5
    saveDir = "$(@__DIR__)/data/"
    xmin = -5
    xmax = 5
    interpAlpha = 6.0

    # Equation
    eq = LinearAdvection(a)

    if j == 1
        randomness = 0.0
        interpRangeConst = 4.5
        method = RK3(MUSCL(1), N, N)
    elseif j == 2
        randomness = (xmax-xmin)/(2*N)
        interpRangeConst = 4.5
        method = RK3(MUSCL(1), N, N)
    elseif j == 3
        randomness = (xmax-xmin)/(2*N)
        interpRangeConst = 3.5
        method = RK3(MUSCL(1), N, N)
    elseif j == 4
        randomness = (xmax-xmin)/(2*N)
        interpRangeConst = 4.5
        method = RK3(CentralGradient(1), N, N)
    elseif j == 5
        randomness = 0.0
        interpRangeConst = 4.5
        method = RK3(CentralGradient(1), N, N)
    elseif j == 6
        randomness = 0.0
        interpRangeConst = 4.5
        method = RK3(MUSCL(2), N, N)
    elseif j == 7
        randomness = (xmax-xmin)/(2*N)
        interpRangeConst = 4.5
        method = RK3(MUSCL(2), N, N)
    elseif j == 8
        randomness = (xmax-xmin)/(2*N)
        interpRangeConst = 3.5
        method = RK3(MUSCL(2), N, N)
    elseif j == 9
        randomness = (xmax-xmin)/(2*N)
        interpRangeConst = 4.5
        method = RK3(CentralGradient(2), N, N)
    elseif j == 10
        randomness = 0.0
        interpRangeConst = 4.5
        method = RK3(CentralGradient(2), N, N)
    elseif j == 11
        randomness = (xmax-xmin)/(2*N)
        interpRangeConst = sqrt(5.0^2 + 3.0^2)+0.01
        method = RK3(DumbserWENO(2), N, N)
    elseif j == 12
        randomness = (xmax-xmin)/(2*N)
        interpRangeConst = sqrt(5.0^2 + 3.0^2)+0.01
        method = RK3(WENO(2), N, N)
    end

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Grid stuff
    dx = (xmax - xmin)/N
    particleGrid = ParticleGrid2D(xmin, xmax, xmin, xmax, N, N; randomness = (dx/2, dx/2))
    setInitialConditions!(particleGrid, smoothInit)
    
    dt = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*(xmax-xmin)/(N))

    # SimSettings
    saveFreq = 100000000000
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRangeConst*(xmax-xmin)/(N),
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq)

    err = Vector{Float64}(undef, length(particleGrid.grid))
    initTimeStep(method.gradientInterpolator, particleGrid, interpAlpha, interpRangeConst*dx)
    rhos = map(particle -> particle.rho, particleGrid.grid)

    for (particleIndex, particle) in enumerate(particleGrid.grid)
        div = method.gradientInterpolator(particleGrid, particleIndex, rhos, eq.vel, settings)
        grad = gradSmoothInit(particle.pos[1], particle.pos[2])
        @assert !isnan(div) "Nan from algorithm $(j)"
        err[particleIndex] = abs(dot(grad, eq.vel) - div)
    end
    
    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
    
    return mean(err)
end

function testConvergence()

    Ns = trunc.(Int64, 10 .^ range(1.0, stop=2.8, length=8))
    algs = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]

    errs = Matrix{Float64}(undef, length(Ns), length(algs))

    for (i, N) in enumerate(Ns)
        for (j, alg) in enumerate(algs)
            errs[i, j] = computeError(alg, N)
        end
    end
    
    p1 = plot(Ns, errs[:, 1], label="MUSCL1 uniform grid (h=4.5dx)", legend=:bottomleft, xscale=:log10, yscale=:log10, ls=:solid, markershape=:circle, xlabel=L"N_x", ylabel="Absolute error", title="Mean error on divergence", size=(900, 600))
    plot!(p1, Ns, errs[:, 2], label="MUSCL1 irregular grid (h=4.5dx)", ls=:dashdot, markershape=:circle)
    plot!(p1, Ns, errs[:, 3], label="MUSCL1 irregular grid (h=3.5dx)", ls=:dashdot, markershape=:circle)
    plot!(p1, Ns, errs[:, 4], label="Central gradient 1 irregular grid (h=4.5dx)", ls=:solid, markershape=:star)
    plot!(p1, Ns, errs[:, 5], label="Central gradient 1 uniform grid (h=4.5dx)", ls=:solid, markershape=:star)
    plot!(p1, Ns, errs[:, 7], label="MUSCL2 irregular grid (h=4.5dx)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 8], label="MUSCL2 irregular grid (h=3.5dx)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 6], label="MUSCL2 uniform grid (h=4.5dx)", ls=:dashdot, markershape=:cross)
    plot!(p1, Ns, errs[:, 9], label="Central gradient 2 irregular grid (h=4.5dx)", ls=:solid, markershape=:square)
    plot!(p1, Ns, errs[:, 10], label="Central gradient 2 uniform grid (h=4.5dx)", ls=:solid, markershape=:square)
    plot!(p1, Ns, errs[:, 11], label="DumbserWENO 2 irregular grid", ls=:solid, markershape=:square)
    plot!(p1, Ns, errs[:, 12], label="WENO 2 irregular grid", ls=:solid, markershape=:square)
    plot!(p1, Ns, 1 ./ Ns, ls=:dash, label="Order 1 reference")
    plot!(p1, Ns, 5 ./ (Ns.^2), ls=:dash, label="Order 2 reference")
    savefig(p1, "$(@__DIR__)/convergence2D.pdf")
end

testConvergence()