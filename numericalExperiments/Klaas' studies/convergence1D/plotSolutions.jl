using Plots
using JLD2
using LaTeXStrings
using Meshfree4ScalarEq.Particles
using Meshfree4ScalarEq.ParticleGrids

# Plot some solutions for the shock initial condition

function shockInit(x::Real)
    return x > 0.0 ? 1.0 : 0.0
end

function smoothInit1(x::Real)
    return exp(-x^2)
end

function smoothInit2(x::Real)
    return sin(2*π*x/5)
end

function mapGrid(x; xmin, xmax)
    temp = x .- xmin
    temp1 = mod.(temp, xmax-xmin)
    return temp1 .+ xmin
end

function plotSolutions(init::String)

    if init == "shockInit"
        initFunc = shockInit
        saveName = "shockSolutions"
        legendLoc = :top
    elseif init == "smoothInit1"
        initFunc = smoothInit1
        saveName = "smoothInit1Solutions"
        legendLoc = :topleft
    elseif init == "smoothInit2"
        initFunc = smoothInit2
        saveName = "smoothInit2Solutions"
        legendLoc = :bottomright
    else
        error(1)
    end

    # Parameters
    tmax = 2.5
    a = 1
    Nx = 100
    ms = 3
    algList = ["RK3MOODMUSCL4_irgrid_$(init)_$(Nx)"; "RK3MOODMUSCL2_irgrid_$(init)_$(Nx)"; "RK3WENO2_irgrid_$(init)_$(Nx)"]

    # Add initial conditions
    d = load("$(@__DIR__)/data/RK3MOODMUSCL4_irgrid_$(init)_$(Nx)/data/step0.jld2")
    particleGrid = d["particleGrid"]
    rhos = map(particle -> particle.rho, particleGrid.grid)
    pos = map(particle -> particle.pos, particleGrid.grid)
    p = plot(pos, rhos, label="Initial condition", seriescolor=:grey, linestyle=:dash)

    # Plot exact solution
    d = load("$(@__DIR__)/data/RK3MOODMUSCL4_irgrid_$(init)_$(Nx)/data/step0.jld2")
    particleGrid = d["particleGrid"]
    xsRegular = particleGrid.xmin:particleGrid.dx:particleGrid.xmax
    xsRegularT = xsRegular .- a*tmax  
    xsRegularT = mapGrid(xsRegularT; xmin=particleGrid.xmin, xmax=particleGrid.xmax)
    exactSolT = initFunc.(xsRegularT)  # Exact solution at t = tmax
    plot!(p, xsRegular, exactSolT, label="Exact solution", xlabel="x", ylabel=L"\rho", c=:blue)
    
    # Plot MUSCL 4
    d = load("$(@__DIR__)/data/$(algList[1])/data/step1.jld2")
    particleGrid1 = d["particleGrid"]
    rhos1 = map(particle -> particle.rho, particleGrid1.grid)
    pos1 = map(particle -> particle.pos, particleGrid1.grid)
    plot!(p, pos1, rhos1, label="MUSCL + MOOD: order 4", c=:maroon1, legend=legendLoc, ls=:dash)
    
    # Plot MUSCL 2
    d = load("$(@__DIR__)/data/$(algList[2])/data/step1.jld2")
    particleGrid2 = d["particleGrid"]
    rhos2 = map(particle -> particle.rho, particleGrid2.grid)
    pos2 = map(particle -> particle.pos, particleGrid2.grid)
    plot!(p, pos2, rhos2, xlabel="x", ylabel=L"\rho", label="MUSCL + MOOD: order 2", c=:gold3, ls=:dashdot)
    
    # Plot WENO 2
    d = load("$(@__DIR__)/data/$(algList[3])/data/step1.jld2")
    particleGrid3 = d["particleGrid"]
    rhos3 = map(particle -> particle.rho, particleGrid3.grid)
    pos3 = map(particle -> particle.pos, particleGrid3.grid)
    plot!(p, pos3, rhos3, xlabel="x", ylabel=L"\rho", label="WENO: order 2", c=:green, ls=:dashdotdot)
    
    # Plot MOOD Events
    plot!(p, [pos2[i] for i in eachindex(pos2) if particleGrid2.grid[i].moodEvent], [rhos2[i] for i in eachindex(pos2) if particleGrid2.grid[i].moodEvent], label="", markershape=:star5, lt=:scatter, c=:gold3, ms=ms)
    plot!(p, [pos1[i] for i in eachindex(pos1) if particleGrid1.grid[i].moodEvent], [rhos1[i] for i in eachindex(pos1) if particleGrid1.grid[i].moodEvent], label="", markershape=:star5, lt=:scatter, c=:maroon1, ms=ms)
    plot!(p, [pos3[i] for i in eachindex(pos3) if particleGrid3.grid[i].moodEvent], [rhos3[i] for i in eachindex(pos3) if particleGrid3.grid[i].moodEvent], label="", markershape=:star5, lt=:scatter, c=:green, ms=ms)

    savefig(p, "$(@__DIR__)/$(saveName).pdf")
end

plotSolutions("shockInit")
plotSolutions("smoothInit1")
plotSolutions("smoothInit2")