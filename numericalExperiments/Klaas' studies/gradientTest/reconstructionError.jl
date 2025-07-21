# Plot convergence of several reconstructions to be used in the MUSCL schemes
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
    xmin = -5
    xmax = 5
    interpAlpha = 1.0
    interpConst = 3.5

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    Ns = 10:100:1000

    configs = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13; 14]
    errs = Matrix{Float64}(undef, length(Ns), length(configs))

    for (i, N) in enumerate(Ns)
        dx = (xmax-xmin)/(N)
        randomness = (xmax-xmin)/(2*N)
        m1 = RK3(MUSCL(1), N)
        m2 = RK3(MUSCL(2), N)
        m3 = RK3(MUSCL(3), N)
        m4 = RK3(MUSCL(4), N)

        # Grid stuff
        particleGrid = ParticleGrid1D(xmin, xmax, N; randomness=randomness)
        updateNeighbours!(particleGrid, interpConst*dx)
        setInitialConditions!(particleGrid, smoothInit)
        rhos = map(particle -> particle.rho, particleGrid.grid)

        # Compute interpolation error using different schemes on the whole grid
        for config in configs
            err = Vector{Float64}(undef, length(particleGrid.grid))
            if (config == 1) || (config == 7)
                initTimeStep(m1.gradientInterpolator, particleGrid, interpAlpha, interpConst*particleGrid.dx)  # MUSCL 1
            elseif (config == 2) || (config == 3) || (config == 8) || (config == 9)
                initTimeStep(m2.gradientInterpolator, particleGrid, interpAlpha, interpConst*particleGrid.dx)  # MUSCL 2
            elseif (config == 4) || (config == 5) || (config == 6) || (config == 10) || (config == 11) || (config == 12)
                initTimeStep(m3.gradientInterpolator, particleGrid, interpAlpha, interpConst*particleGrid.dx)  # MUSCL 3
            elseif (config == 13) || (config == 14)
                initTimeStep(m4.gradientInterpolator, particleGrid, interpAlpha, interpConst*particleGrid.dx)  # MUSCL 4
            end
            for (particleIndex, particle) in enumerate(particleGrid.grid)
                # Interpolate halfway to the first neighbour (fi1)
                nbIndex = particle.neighbourIndices[1]
                nbParticle = particleGrid.grid[nbIndex]
                deltaPos = getPeriodicDistance(particleGrid, particleIndex, nbIndex)

                if config == 1  # Linear reconstruction using MUSCL 1
                    fi1 = rhos[particleIndex] + 0.5*deltaPos*sum(particle.alfaij[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
                elseif config == 2  # Linear reconstruction using MUSCL 2
                    fi1 = rhos[particleIndex] + 0.5*deltaPos*sum(particle.alfaijBar[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
                elseif config == 3  # Quadratic reconstruction using MUSCL 2
                    fi1 = rhos[particleIndex] + 0.5*deltaPos*sum(particle.alfaijBar[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
                    fi1 += (deltaPos^2)*sum(particle.betaij[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/8
                elseif config == 4  # Linear reconstruction using MUSCL 3
                    fi1 = rhos[particleIndex] + 0.5*deltaPos*sum(particle.alfaijBar[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
                elseif config == 5  # Quadratic reconstruction using MUSCL 3
                    fi1 = rhos[particleIndex] + 0.5*deltaPos*sum(particle.alfaijBar[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
                    fi1 += (deltaPos^2)*sum(particle.betaij[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/8
                elseif config == 6  # Cubic reconstruction using MUSCL 3
                    fi1 = rhos[particleIndex] + 0.5*deltaPos*sum(particle.alfaijBar[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
                    fi1 += (deltaPos^2)*sum(particle.betaij[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/8
                    fi1 += (deltaPos^3)*sum(particle.alfaij[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/(6*8)
                elseif config == 7  # Linear reconstruction using MUSCL 1
                    fi1 = rhos[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaij[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
                elseif config == 8  # Linear reconstruction using MUSCL 2
                    fi1 = rhos[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaijBar[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
                elseif config == 9  # Quadratic reconstruction using MUSCL 2
                    fi1 = rhos[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaijBar[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
                    fi1 += (deltaPos^2)*sum(nbParticle.betaij[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/8
                elseif config == 10  # Linear reconstruction using MUSCL 3
                    fi1 = rhos[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaijBar[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
                elseif config == 11  # Quadratic reconstruction using MUSCL 3
                    fi1 = rhos[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaijBar[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
                    fi1 += (deltaPos^2)*sum(nbParticle.betaij[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/8
                elseif config == 12  # Cubic reconstruction using MUSCL 3
                    fi1 = rhos[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaijBar[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
                    fi1 += (deltaPos^2)*sum(nbParticle.betaij[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/8
                    fi1 -= (deltaPos^3)*sum(nbParticle.alfaij[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/(6*8)
                elseif config == 13  # Quartic reconstruction using MUSCL 4
                    fi1 = rhos[particleIndex] + 0.5*deltaPos*sum(particle.alfaijBar[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
                    fi1 += (deltaPos^2)*sum(particle.betaij[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/8
                    fi1 += (deltaPos^3)*sum(particle.alfaij[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/(6*8)
                    fi1 += (deltaPos^4)*sum(particle.gammaij[i]*(rhos[k] - rhos[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/(24*(2^4))
                elseif config == 14  # Quartic reconstruction using MUSCL 4
                    fi1 = rhos[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaijBar[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
                    fi1 += (deltaPos^2)*sum(nbParticle.betaij[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/8
                    fi1 -= (deltaPos^3)*sum(nbParticle.alfaij[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/(6*8)
                    fi1 += (deltaPos^4)*sum(nbParticle.gammaij[i]*(rhos[k] - rhos[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/(24*(2^4))
                    end
                err[particleIndex] = abs(smoothInit(particle.pos + deltaPos/2) - fi1)
            end
            errs[i, config] = mean(err)
        end
    end

    p1 = plot(Ns, errs[:, 1], label="MUSCL1 linear rec. (i)", legend=:bottomleft, xscale=:log10, yscale=:log10, ls=:solid, markershape=:circle, xlabel=L"N_x", ylabel="Absolute error", title="Reconstruction error", size=(900, 600))
    plot!(p1, Ns, errs[:, 2], label="MUSCL2 linear rec. (i)", ls=:solid, markershape=:circle)
    plot!(p1, Ns, errs[:, 3], label="MUSCL2 quadratic rec. (i)", ls=:solid, markershape=:circle)
    plot!(p1, Ns, errs[:, 4], label="MUSCL3 linear rec. (i)", ls=:solid, markershape=:circle)
    plot!(p1, Ns, errs[:, 5], label="MUSCL3 quadratic rec. (i)", ls=:solid, markershape=:circle)
    plot!(p1, Ns, errs[:, 6], label="MUSCL3 cubic rec. (i)", ls=:solid, markershape=:circle)
    plot!(p1, Ns, errs[:, 7], label="MUSCL1 linear rec. (j)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 8], label="MUSCL2 linear rec. (j)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 9], label="MUSCL2 quadratic rec. (j)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 10], label="MUSCL3 linear rec. (j)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 11], label="MUSCL3 quadratic rec. (j)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 12], label="MUSCL3 cubic rec. (j)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 13], label="MUSCL4 quartic. (i)", ls=:solid, markershape=:cross)
    plot!(p1, Ns, errs[:, 14], label="MUSCL4 quartic. (j)", ls=:solid, markershape=:cross)

    plot!(p1, Ns, 1 ./ Ns, ls=:dash, label="Order 1 reference")
    plot!(p1, Ns, 10 ./ (Ns.^2), ls=:dash, label="Order 2 reference")
    plot!(p1, Ns, 100 ./ (Ns.^3), ls=:dash, label="Order 3 reference")
    plot!(p1, Ns, 300 ./ (Ns.^4), ls=:dash, label="Order 4 reference")
    plot!(p1, Ns, 300 ./ (Ns.^5), ls=:dash, label="Order 5 reference")
    savefig(p1, "$(@__DIR__)/reconstructionError.pdf")

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
end

testConvergence()