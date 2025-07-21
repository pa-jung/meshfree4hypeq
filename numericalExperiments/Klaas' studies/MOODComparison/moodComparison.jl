using Plots
using JLD2
using LaTeXStrings
using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq

function smoothInit1(x::Real)
    return exp(-x^2)
end

function smoothInit2(x::Real)
    return sin(2*π*x/5)
end

function shockInit(x::Real)
    return x > 0.0 ? 1.0 : 0.0
end

function mapGrid(x; xmin, xmax)
    temp = x .- xmin
    temp1 = mod.(temp, xmax-xmin)
    return temp1 .+ xmin
end

const tmax = 2.5
const N = 100
const a = 1.0

function doSimluation(regular::Bool, initFunc::String, AlgID::Integer)
    # Simulation settings
    CFL = 1/3
    xmin = -5
    xmax = 5
    saveDir = "$(@__DIR__)/data/"
    interpAlpha = 6.0
    saveFreq = 10000

    state = copy(Meshfree4ScalarEq.rng)

    # Simlation algorithms
    if AlgID == 1
        method = RK3(MUSCL(2), N; mood = MOODu2(deltaRelax=true))
        saveDir *= "RK3MOODu2MUSCL2"
    elseif AlgID == 2
        method = RK3(MUSCL(2), N; mood = MOODLoubertU2(deltaRelax=true))
        saveDir *= "RK3MOODLoubertMUSCL2"
    elseif AlgID == 3
        method = RK3(MUSCL(2), N; mood = MOODu1(deltaRelax=true))
        saveDir *= "RK3MOODu1MUSCL2"
    elseif AlgID == 4
        method = RK3(MUSCL(2), N; mood = NoMOOD())
        saveDir *= "RK3MUSCL2"
    end

    # Equation
    eq = LinearAdvection(a)
    
    # Grid stuff
    if regular
        particleGrid = ParticleGrid1D(xmin, xmax, N)
        saveDir *= "_rgrid"
    else
        dx = (xmax-xmin)/N
        particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = dx/2)
        saveDir *= "_irgrid"
    end
    interpRange = 3.5*particleGrid.dx
    dt = CFL*getTimeStep(particleGrid, eq, interpAlpha, interpRange)  # Time step such that meshfree first order least squares method is positive. (Check MeshfreeUpwind_irgrid_shockInit)

    # Initial condition
    if initFunc == "smoothInit1"
        setInitialConditions!(particleGrid, smoothInit1)
        saveDir *= "_smoothInit1"
    elseif initFunc == "smoothInit2"
        setInitialConditions!(particleGrid, smoothInit2)
        saveDir *= "_smoothInit2"
    elseif initFunc == "shockInit"
        setInitialConditions!(particleGrid, shockInit)
        saveDir *= "_shockInit"
    end

    # Create SimSettings object
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRange,
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq)

    # Do simulation
    mainTimeIntegrator!(method, eq, particleGrid, settings)

    # Generate plots
    plotDensity(settings, settings.currentSaveNb-1; saveFigure=true, showMOODEvents=true)

    copy!(Meshfree4ScalarEq.rng, state)
end

function plotSolutions(init)
    doSimluation(false, init, 1)
    doSimluation(false, init, 2)
    doSimluation(false, init, 3)
    doSimluation(false, init, 4)

    # Add initial conditions
    d = load("$(@__DIR__)/data/RK3MOODu1MUSCL2_irgrid_$(init)/data/step0.jld2")
    particleGrid = d["particleGrid"]
    rhos = map(particle -> particle.rho, particleGrid.grid)
    pos = map(particle -> particle.pos, particleGrid.grid)
    p = plot(pos, rhos, label="Initial condition", seriescolor=:grey, linestyle=:dash, legend=:bottom)

    # Plot exact solution
    d = load("$(@__DIR__)/data/RK3MOODu1MUSCL2_irgrid_$(init)/data/step0.jld2")
    particleGrid = d["particleGrid"]
    xsRegular = particleGrid.xmin:particleGrid.dx:particleGrid.xmax
    xsRegularT = xsRegular .- a*tmax  
    xsRegularT = mapGrid(xsRegularT; xmin=particleGrid.xmin, xmax=particleGrid.xmax)
    if init == "shockInit"
        exactSolT = shockInit.(xsRegularT)  # Exact solution at t = tmax
    elseif init == "smoothInit1"
        exactSolT = smoothInit1.(xsRegularT)  # Exact solution at t = tmax
    elseif init == "smoothInit2"
        exactSolT = smoothInit2.(xsRegularT)  # Exact solution at t = tmax
    end

    plot!(p, xsRegular, exactSolT, label="Exact solution", xlabel="x", ylabel=L"\rho")
    
    # Plot NoMOOD
    d = load("$(@__DIR__)/data/RK3MUSCL2_irgrid_$(init)/data/step1.jld2")
    particleGrid1 = d["particleGrid"]
    rhos1 = map(particle -> particle.rho, particleGrid1.grid)
    pos1 = map(particle -> particle.pos, particleGrid1.grid)
    plot!(p, pos1, rhos1, label="No MOOD", c=:maroon1, ls=:solid)
    
    # Plot MOOD u1
    d = load("$(@__DIR__)/data/RK3MOODu1MUSCL2_irgrid_$(init)/data/step1.jld2")
    particleGrid2 = d["particleGrid"]
    rhos2 = map(particle -> particle.rho, particleGrid2.grid)
    pos2 = map(particle -> particle.pos, particleGrid2.grid)
    plot!(p, pos2, rhos2, xlabel="x", ylabel=L"\rho", label="u1", c=:gold3, ls=:solid)
    
    # Plot MOOD Loubert
    d = load("$(@__DIR__)/data/RK3MOODLoubertMUSCL2_irgrid_$(init)/data/step1.jld2")
    particleGrid3 = d["particleGrid"]
    rhos3 = map(particle -> particle.rho, particleGrid3.grid)
    pos3 = map(particle -> particle.pos, particleGrid3.grid)
    plot!(p, pos3, rhos3, xlabel="x", ylabel=L"\rho", label="Loubert u2", c=:green, ls=:dashdotdot)
    
    # Plot MOOD u2
    d = load("$(@__DIR__)/data/RK3MOODu2MUSCL2_irgrid_$(init)/data/step1.jld2")
    particleGrid4 = d["particleGrid"]
    rhos4 = map(particle -> particle.rho, particleGrid4.grid)
    pos4 = map(particle -> particle.pos, particleGrid4.grid)
    plot!(p, pos4, rhos4, xlabel="x", ylabel=L"\rho", label="Corrected u2", c=:blue, ls=:dashdot)

    # Plot MOODu2
    ms = 3
    plot!(p, [pos2[i] for i in eachindex(pos2) if particleGrid2.grid[i].moodEvent], [rhos2[i] for i in eachindex(pos2) if particleGrid2.grid[i].moodEvent], label="", markershape=:star5, lt=:scatter, c=:gold3, ms=ms)
    plot!(p, [pos3[i] for i in eachindex(pos3) if particleGrid3.grid[i].moodEvent], [rhos3[i] for i in eachindex(pos3) if particleGrid3.grid[i].moodEvent], label="", markershape=:star5, lt=:scatter, c=:green, ms=ms)
    plot!(p, [pos4[i] for i in eachindex(pos4) if particleGrid4.grid[i].moodEvent], [rhos4[i] for i in eachindex(pos3) if particleGrid4.grid[i].moodEvent], label="", markershape=:star5, lt=:scatter, c=:blue, ms=ms)

    savefig(p, "$(@__DIR__)/moodComparison_$(init).pdf")
    

end

plotSolutions("shockInit")
plotSolutions("smoothInit1")
plotSolutions("smoothInit2")