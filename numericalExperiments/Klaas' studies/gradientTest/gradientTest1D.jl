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

function smoothInit(x::Real)
    return sin(x*pi/5)
end

function dSmoothInit(x::Real)
    return cos(x*pi/5)*pi/5
end

function testConvergence()
    tmax = 7.5
    a = 1.0;
    saveDir = "$(@__DIR__)/data/"
    xmin = -5
    xmax = 5
    interpAlpha = 1.0

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Equation
    eq = LinearAdvection(a)

    Ns = 10:100:1000
    randomnessAr = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13]

    errs = Matrix{Float64}(undef, length(Ns), length(randomnessAr))

    for (i, N) in enumerate(Ns)
        for j in randomnessAr
            if j == 1
                randomness = 0.0
                interpRangeConst = 3.5
                method = RK3(MUSCL(1), N)
            elseif j == 2
                randomness = (xmax-xmin)/(2*N)
                interpRangeConst = 3.5
                method = RK3(MUSCL(1), N)
            elseif j == 3
                randomness = (xmax-xmin)/(2*N)
                interpRangeConst = 1.5
                method = RK3(MUSCL(1), N)
            elseif j == 4
                randomness = (xmax-xmin)/(2*N)
                interpRangeConst = 3.5
                method = RK3(CentralGradient(1), N)
            elseif j == 5
                randomness = 0.0
                interpRangeConst = 3.5
                method = RK3(CentralGradient(1), N)
            elseif j == 6
                randomness = 0.0
                interpRangeConst = 3.5
                method = RK3(MUSCL(2), N)
            elseif j == 7
                randomness = (xmax-xmin)/(2*N)
                interpRangeConst = 3.5
                method = RK3(MUSCL(2), N)
            elseif j == 8
                randomness = (xmax-xmin)/(2*N)
                interpRangeConst = 1.5
                method = RK3(MUSCL(2), N)
            elseif j == 9
                randomness = (xmax-xmin)/(2*N)
                interpRangeConst = 3.5
                method = RK3(CentralGradient(2), N)
            elseif j == 10
                randomness = 0.0
                interpRangeConst = 3.5
                method = RK3(CentralGradient(2), N)
            elseif j == 11
                randomness = (xmax-xmin)/(2*N)
                interpRangeConst = 3.5
                method = RK3(MUSCL(3), N)
            elseif j == 12
                randomness = (xmax-xmin)/(2*N)
                interpRangeConst = 3.5
                method = RK3(MUSCL(4), N)
            elseif j == 13
                randomness = (xmax-xmin)/(2*N)
                interpRangeConst = 3.5
                method = RK3(WENO(2), N)
            end
            # Grid stuff
            particleGrid = ParticleGrid1D(xmin, xmax, N; randomness=randomness)
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
            initTimeStep(method.gradientInterpolator, particleGrid, interpAlpha, interpRangeConst*(xmax-xmin)/(N))
            rhos = map(particle -> particle.rho, particleGrid.grid)
            if (j != 9) && (j != 10) && (j != 4) && (j != 5)
                for (particleIndex, particle) in enumerate(particleGrid.grid)
                    res = method.gradientInterpolator(particleGrid, particleIndex, rhos, eq.vel, settings)/eq.vel
                    err[particleIndex] = abs(dSmoothInit(particle.pos) - res)
                end
            end

            errs[i, j] = mean(err)
        end
    end

    p1 = plot(Ns, errs[:, 1], label="MUSCL1 uniform grid (h=3.5dx)", legend=:bottomleft, xscale=:log10, yscale=:log10, ls=:solid, markershape=:circle, xlabel=L"N_x", ylabel="Absolute error", title="Mean error on gradient", size=(900, 600))
    plot!(p1, Ns, errs[:, 2], label="MUSCL1 irregular grid (h=3.5dx)", ls=:solid, markershape=:circle)
    plot!(p1, Ns, errs[:, 3], label="MUSCL1 irregular grid (h=1.5dx)", ls=:solid, markershape=:circle)
    # plot!(p1, Ns, errs[:, 4], label="Central gradient 1 irregular grid (h=3.5dx)", ls=:solid, markershape=:star)
    # plot!(p1, Ns, errs[:, 5], label="Central gradient 1 uniform grid (h=3.5dx)", ls=:solid, markershape=:star)
    plot!(p1, Ns, errs[:, 6], label="MUSCL2 uniform grid (h=3.5dx)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 7], label="MUSCL2 irregular grid (h=3.5dx)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 8], label="MUSCL2 irregular grid (h=1.5dx)", ls=:solid, markershape=:cross)
    # plot!(p1, Ns, errs[:, 9], label="Central gradient 2 irregular grid (h=3.5dx)", ls=:solid, markershape=:square)
    # plot!(p1, Ns, errs[:, 10], label="Central gradient 2 uniform grid (h=3.5dx)", ls=:solid, markershape=:square)
    plot!(p1, Ns, errs[:, 11], label="MUSCL3 gradient irregular (h=3.5dx)", ls=:solid, markershape=:square)
    plot!(p1, Ns, errs[:, 12], label="MUSCL4 gradient irregular (h=3.5dx)", ls=:solid, markershape=:square)
    plot!(p1, Ns, errs[:, 13], label="WENO2 gradient irregular (h=3.5dx)", ls=:solid, markershape=:square)
    plot!(p1, Ns, 1 ./ Ns, ls=:dash, label="Order 1 reference")
    plot!(p1, Ns, 10 ./ (Ns.^2), ls=:dash, label="Order 2 reference")
    plot!(p1, Ns, 30 ./ (Ns.^3), ls=:dash, label="Order 3 reference")
    plot!(p1, Ns, 30 ./ (Ns.^4), ls=:dash, label="Order 4 reference")
    savefig(p1, "$(@__DIR__)/convergence1D.pdf")

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
end

testConvergence()
