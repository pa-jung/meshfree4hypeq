module ParticleGrids

using FileIO, JLD2
using Plots
using Printf
using LaTeXStrings
using Statistics
using VoronoiCells
using GeometryBasics
using LinearAlgebra
using ..Particles
using ..SimSettings
using ..ScalarHyperbolicEquations
import Meshfree4ScalarEq

export ParticleGrid, ParticleGrid1D, ParticleGrid2D, setInitialConditions!, getPeriodicDistance, saveGrid, plotDensity, animateDensity, getTimeStep, findLocalExtrema!, updateVoxelInformation!
export gridToLinearIndex, linearIndexToGrid, findNeighbouringVoxels, updateNeighbours!, getEuclideanDistance, logMOODEvents!, findLocalExtremaAbs!

abstract type ParticleGrid end

struct ParticleGrid1D <: ParticleGrid
    grid::Vector{Particle1D}  # Vector containing Particle objects
    xmin::Float64  # Min & max of domain. Periodic domain is assumed. Only a particle at xmin is stored.
    xmax::Float64
    N::Int64  # Amount of particles in grid (= length(grid))
    dx::Float64  # (Average) grid spacing 
    regular::Bool  # Bool to indiciate if grid is regular or not
    temp::Vector{Float64}  # Used for storing curvatures during time integration

    """
        ParticleGrid1D(xmin<:Real, xmax<:Real, N::Integer; randomness<:Real)

    Construct a ParticleGrid object. This function will generate a periodic grid (particle at the right edge not stored) of N points in [xmin, xmax).

    -   An unstructured grid can be created by setting randomness (uniform variance on position). Particle points are then generated as x_{k+1} = x_k + dx + randomness*(rand(rng, Float64)*2 - 1)
    -   Regular grids can be created by setting randomnes to 0.0
    -   A unstructured grid can also be created by setting randomnes to something negative. In that case, the grid is created by sorting rand(N) numbers between xmin and xmax.
    -   If singlePointPerturbation = true, the randomness is a perturbation of the center point.
    """
    function ParticleGrid1D(xmin::Real, xmax::Real, N::Integer; randomness::Real = 0.0, rng = Meshfree4ScalarEq.rng, singlePointPerturbation::Bool=false)
        @assert N >= 2
        @assert xmax > xmin
        dx = (xmax-xmin)/N

        # Create grid
        grid = Vector{Particle1D}(undef, N)
        if singlePointPerturbation
            for i = 1:N
                if i == round(N/2)  # perturb center point
                    pos = xmin + dx*(i-0.5) + randomness
                else
                    pos = xmin + dx*(i-0.5)
                end
                grid[i] = Particle1D(pos, 0.0, false)
            end
        else
            if randomness >= 0.0
                @assert randomness <= 0.5*dx "Randomness too large. Faulty grids could be generated."
                for i = 1:N
                    pos = xmin + dx*(i-0.5) + randomness*(rand(rng, Float64)*2 - 1)
                    grid[i] = Particle1D(pos, 0.5, false)
                end
            else
                for i = 1:N
                    pos = prevfloat(convert(Float64, xmax-xmin))*(1-rand(rng, Float64)) + xmin  # Random uniform number in (xmin, xmax)
                    grid[i] = Particle1D(pos, 0.0, false)
                end
                sort!(grid, by=particle->particle.pos)
            end
        end
        @assert all(map(particle->particle.pos, grid) .>= xmin)
        @assert all(map(particle->particle.pos, grid) .<= xmax)

        regular = randomness == 0.0 ? true : false

        # Assign volume
        for (particleIndex, particle) in enumerate(grid)
            if particleIndex == 1
                deltaPosL = (particle.pos - xmin) + (xmax - grid[N].pos)
            else
                deltaPosL = particle.pos - grid[particleIndex - 1].pos 
            end
            if particleIndex == N
                deltaPosR = (xmax - particle.pos) + (grid[1].pos - xmin)
            else
                deltaPosR = grid[particleIndex + 1].pos - particle.pos
            end
            particle.volume = (deltaPosL + deltaPosR)/2
        end

        new(grid, convert(Float64, xmin), convert(Float64, xmax), convert(Int64, N), dx, regular, Vector{Float64}(undef, N))
    end

    """
        ParticleGrid1D(grid::Vector{Particle1D}, xmin<:Real, xmax<:Real)
    
    The default constructor. 
    """
    function ParticleGrid1D(grid::Vector{Particle1D}, xmin::Real, xmax::Real, dx::Real, regular::Bool)
        @assert length(grid) >= 2
        @assert issorted(grid, by=particle->particle.pos) "Grid is not sorted."
        @assert count(map(particle -> particle.boundary, grid)) == 1 "Only the leftmost particle is allowed to be a boundary particle."
        new(grid, convert(Float64, xmin), convert(Float64, xmax), length(grid), convert(Float64, dx), regular, Vector{Float64}(undef, length(grid)))
    end
end

struct ParticleGrid2D <: ParticleGrid
    grid::Vector{Particle2D}
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    Nx::Int64
    Ny::Int64
    dx::Float64
    dy::Float64
    regular::Bool
    temp::Matrix{Float64}

    """
        ParticleGrid2D(xmin::Real, xmax::Real, ymin::Real, ymax::Real, Nx::Integer, Ny::Integer; randomness::Tuple{Real, Real} = 0.0)

    Construct a ParticleGrid object. This function will generate a 2D periodic grid (particles at the right edge not stored) of Nx x Ny points in [xmin, xmax) x [ymin, ymax).
    An unstructured grid can be created by setting randomness (uniform variance on position) to something positive.
    """
    function ParticleGrid2D(xmin::Real, xmax::Real, ymin::Real, ymax::Real, Nx::Integer, Ny::Integer; randomness::Tuple{Real, Real} = (0.0, 0.0), rng = Meshfree4ScalarEq.rng)
        @assert Nx >= 2
        @assert Ny >= 2
        @assert xmax > xmin
        @assert ymax > ymin

        grid = Vector{Particle2D}(undef, Nx*Ny)

        dx = (xmax - xmin)/Nx
        dy = (ymax - ymin)/Ny

        # Create grid
        index = 1
        for i = 1:Nx
            for j = 1:Ny
                boundary = (i == 1) || (j == 1) ? true : false
                posX = xmin + dx*(i-0.5) + randomness[1]*(rand(rng, Float64)*2 - 1) 
                posY = ymin + dy*(j-0.5) + randomness[2]*(rand(rng, Float64)*2 - 1)
                grid[index] = Particle2D((posX, posY), 0.0, boundary)
                index += 1
                @assert (xmin <= posX <= xmax) && (ymin <= posY <= ymax)
            end
        end

        regular = (randomness[1] == 0.0) && (randomness[2] == 0.0) ? true : false

        # Set volumes
        points = [Point2(particle.pos) for particle in grid]
        rect = Rectangle(Point2(xmin-dx, ymin-dy), Point2(xmax+dx, ymax+dy))
        tess = voronoicells(points, rect)
        vols = voronoiarea(tess)
        for (particleIndex, particle) in enumerate(grid)
            particle.volume = vols[particleIndex]
        end
        
        new(grid, convert(Float64, xmin), convert(Float64, xmax), convert(Float64, ymin), convert(Float64, ymax), convert(Int64, Nx), convert(Int64, Ny), dx, dy, regular, Matrix{Float64}(undef, Nx*Ny, 2))
    end
end


"""
    getTimeStep(particleGrid::ParticleGrid1D, eq::LinearAdvection, interpAlpha::Real, interpRange::Real)

Return the maximum time step for which the first-order euler & upwind method is a positive scheme (for linear advection equation).
"""
function getTimeStep(particleGrid::ParticleGrid1D, eq::LinearAdvection, interpAlpha::Real, interpRange::Real)
    dtMax = Inf
    updateNeighbours!(particleGrid, interpRange)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        num = 0.0
        denum = 0.0
        for nbIndex in particle.neighbourIndices
            dx = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
            if ((eq.vel >= 0.0) && (dx <= 0.0)) || ((eq.vel <= 0.0) && (dx >= 0.0))
                w = exp(-interpAlpha*((dx)^2))
                num += w*dx
                denum += w*dx*dx
            end
        end
        dtMax = min(-denum/(eq.vel*num), dtMax)
    end
    return dtMax
end

"""
    getTimeStep(particleGrid::ParticleGrid2D, eq::LinearAdvection, interpAlpha::Real, interpRange::Real)

Return the maximum time step for which Praveen's upwind method is a positive scheme (for linear advection equation).
"""
function getTimeStep(particleGrid::ParticleGrid2D, eq::LinearAdvection, interpAlpha::Real, interpRange::Real)
    dtMax = Inf
    updateNeighbours!(particleGrid, interpRange)

    for (particleIndex, particle) in enumerate(particleGrid.grid)
        # Create 2x2 LS system
        A11 = A12 = A22 = 0.0
        for nbIndex in particle.neighbourIndices
            deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
            w = exp(-interpAlpha*(deltaX^2 + deltaY^2)/(interpRange^2))
            A11 += w*(deltaX^2) 
            A12 += w*deltaX*deltaY
            A22 += w*(deltaY^2)
        end
        D = A11*A22 - (A12^2)
        
        sumCij = 0.0
        for nbIndex in particle.neighbourIndices

            # Solve 2x2 LS system
            deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
            w = exp(-interpAlpha*(deltaX^2 + deltaY^2)/(interpRange^2))
            coeff = ((A22*w*deltaX - A12*w*deltaY)/D, (A11*w*deltaY - A12*w*deltaX)/D)

            # Compute adapted coefficients
            angle = atan(deltaY, deltaX)  # atan2 function
            n = (cos(angle), sin(angle))
            s = (-sin(angle), cos(angle))
            alfaBar = dot(n, coeff)
            betaBar = dot(s, coeff)
            bracketMinus = dot(eq.vel, n) > 0.0 ? 0.0 : dot(eq.vel, n)
            bracketMinus2 = betaBar*dot(eq.vel, s) > 0.0 ? 0.0 : betaBar*dot(eq.vel, s)
            sumCij -= alfaBar*bracketMinus + bracketMinus2
        end
        dtMax = min(1/(2*sumCij), dtMax)
    end
    return dtMax
end

"""
    setInitialConditions!(particleGrid::ParticleGrid1D, initFunc::Function)

Given initFunc, a function that takes a Real number and returns the initial value at that point, this function will populate paricleGrid.grid[i].rho.
"""
function setInitialConditions!(particleGrid::ParticleGrid1D, initFunc::Function)
    for i in eachindex(particleGrid.grid)
        particleGrid.grid[i].rho = initFunc(particleGrid.grid[i].pos)
    end
end

"""
    setInitialConditions!(particleGrid::ParticleGrid2D, initFunc::Function)

Given initFunc, a function that takes an x and y coordinate and returns the initial value at that point, this function will populate paricleGrid.grid[i].rho.
"""
function setInitialConditions!(particleGrid::ParticleGrid2D, initFunc::Function)
    for i in eachindex(particleGrid.grid)
        particleGrid.grid[i].rho = initFunc(particleGrid.grid[i].pos[1], particleGrid.grid[i].pos[2])
    end
end

function setInitialConditions!(particleGrids::Vector{T}, initFuncs::Vector{Function}) where T <: ParticleGrid
    @assert length(particleGrids) == length(initFuncs) "For each equation a initial condition has to be given!"
    for i = eachindex(particleGrids)
        setInitialConditions!(particleGrids[i], initFuncs[i])
    end
end

"""
    getPeriodicDistance(particleGrid::ParticleGrid1D, particleIndex::Integer, nbParticle::Integer)

Return the distance from particleIndex to nbParticle taking into account that particles can be neighbours accros the periodic boundary.
If nbParticle is to the right of particleIndex, the distance is postive, and vice versa.
"""
function getPeriodicDistance(particleGrid::ParticleGrid1D, particleIndex::Integer, nbParticle::Integer)
    domainSize = particleGrid.xmax - particleGrid.xmin
    dist = particleGrid.grid[nbParticle].pos - particleGrid.grid[particleIndex].pos
    return dist - round(dist/domainSize)*domainSize
end

"""
    getPeriodicDistance(particleGrid::ParticleGrid2D, particleIndex::Integer, nbParticle::Integer)

Return the distance in x and y from particleIndex to nbParticle taking into account that particles can be neighbours accros the periodic boundary.
"""
function getPeriodicDistance(particleGrid::ParticleGrid2D, particleIndex::Integer, nbParticle::Integer)::Tuple{Float64, Float64}
    domainSizeX = particleGrid.xmax - particleGrid.xmin
    distX = particleGrid.grid[nbParticle].pos[1] - particleGrid.grid[particleIndex].pos[1]
    domainSizeY = particleGrid.ymax - particleGrid.ymin
    distY = particleGrid.grid[nbParticle].pos[2] - particleGrid.grid[particleIndex].pos[2]
    return (distX - round(distX/domainSizeX)*domainSizeX, distY - round(distY/domainSizeY)*domainSizeY)
end

function getEuclideanDistance(particleGrid::ParticleGridType, particleIndex::Integer, nbParticle::Integer) where {ParticleGridType <: ParticleGrid}
    if ParticleGridType == ParticleGrid1D
        return abs(getPeriodicDistance(particleGrid, particleIndex, nbParticle))
    elseif ParticleGridType == ParticleGrid2D
        dx, dy = getPeriodicDistance(particleGrid, particleIndex, nbParticle)
        return sqrt(dx^2 + dy^2)
    else
        error("Not implemented.")
    end
end

"""
    updateNeighbours!(particleGrid::ParticleGrid1D, maxDist<:Real)

For each particle, find the other particles within a distance maxDist and add them to neighbourIndices.
"""
function updateNeighbours!(particleGrid::ParticleGrid1D, maxDist::Real)
    for particleIndex in eachindex(particleGrid.grid)
        particle = particleGrid.grid[particleIndex]
        empty!(particle.neighbourIndices)
        for i in particleIndex-1:-1:particleIndex-1-div(particleGrid.N, 2)  # Walk left & find particles that are within maxDist
            particleLeftIndex = mod1(i, particleGrid.N)
            dist = getEuclideanDistance(particleGrid, particleIndex, particleLeftIndex)
            if dist < maxDist
                push!(particle.neighbourIndices, particleLeftIndex)
            else
                break
            end
        end
        for i in particleIndex+1:particleIndex+1+div(particleGrid.N, 2)  # Walk right & find particles that are within maxDist
            particleRightIndex = mod1(i, particleGrid.N)
            dist = getEuclideanDistance(particleGrid, particleIndex, particleRightIndex)
            if dist < maxDist
                push!(particle.neighbourIndices, particleRightIndex)
            else
                break
            end
        end
        resize!(particle.alfaij, length(particle.neighbourIndices))
        resize!(particle.alfaijBar, length(particle.neighbourIndices))
        resize!(particle.betaij, length(particle.neighbourIndices))
        resize!(particle.gammaij, length(particle.neighbourIndices))
        resize!(particle.dxVec, length(particle.neighbourIndices))
        resize!(particle.wVec, length(particle.neighbourIndices))
        resize!(particle.dfVec, length(particle.neighbourIndices))
        particle.A = Matrix{Float64}(undef, length(particle.neighbourIndices), 4)
    end
end

"""
    gridToLinearIndex(hBox::Integer, vBox::Integer, nbBoxesX::Integer)::Integer

Convert 2D grid index to a linear index.
"""
function gridToLinearIndex(hBox::Integer, vBox::Integer, nbBoxesX::Integer)::Integer
    return hBox + nbBoxesX*vBox
end


"""
    linearIndexToGrid(linearIndex::Integer, nbBoxesX::Integer)::Tuple{Integer, Integer}
    
Convert linear index to row and column index (hBox, vBox).
"""
function linearIndexToGrid(linearIndex::Integer, nbBoxesX::Integer)::Tuple{Integer, Integer}
    hBox = mod(linearIndex, nbBoxesX)
    vBox = div(linearIndex - hBox, nbBoxesX)
    return (hBox, vBox)
end

"""
    updateVoxelInformation!(particleGrid::ParticleGrid2D, maxDist::Real)

Sort the particles into boxes of size maxDist x maxDist.
"""
function updateVoxelInformation!(particleGrid::ParticleGrid2D, maxDist::Real)
    nbBoxesX = floor(Int64, (particleGrid.xmax - particleGrid.xmin)/maxDist)
    nbBoxesY = floor(Int64, (particleGrid.ymax - particleGrid.ymin)/maxDist)
    xBoxSize = (particleGrid.xmax - particleGrid.xmin)/nbBoxesX
    yBoxSize = (particleGrid.ymax - particleGrid.ymin)/nbBoxesY
    
    for particle in particleGrid.grid
        if particle.pos[1] != particleGrid.xmax
            hBox = floor(Int64, (particle.pos[1] - particleGrid.xmin)/xBoxSize)
        else
            hBox = nbBoxesX - 1
        end
        if particle.pos[2] != particleGrid.ymax
            vBox = floor(Int64, (particle.pos[2] - particleGrid.ymin)/yBoxSize)
        else
            vBox = nbBoxesY - 1
        end
        particle.voxel = gridToLinearIndex(hBox, vBox, nbBoxesX)
    end
end


"""
    findNeighbouringVoxels(linearIndex::Integer, nbBoxesX::Integer, nbBoxesY::Integer)::Tuple{Integer, Integer, Integer, Integer, Integer, Integer, Integer, Integer, Integer}

Given the linear index of a box, return the linearIndices of the boxes in which a neighbouring particle can reside (includes voxel linearIndex).
"""
function findNeighbouringVoxels(linearIndex::Integer, nbBoxesX::Integer, nbBoxesY::Integer)::Tuple{Integer, Integer, Integer, Integer, Integer, Integer, Integer, Integer, Integer}
    xBox, yBox = linearIndexToGrid(linearIndex, nbBoxesX)
    i1 = gridToLinearIndex(mod(xBox+1, nbBoxesX), yBox, nbBoxesX)
    i2 = gridToLinearIndex(mod(xBox+1, nbBoxesX), mod(yBox+1, nbBoxesY), nbBoxesX)
    i3 = gridToLinearIndex(mod(xBox+1, nbBoxesX), mod(yBox-1, nbBoxesY), nbBoxesX)
    i4 = gridToLinearIndex(mod(xBox-1, nbBoxesX), yBox, nbBoxesX)
    i5 = gridToLinearIndex(mod(xBox-1, nbBoxesX), mod(yBox+1, nbBoxesY), nbBoxesX)
    i6 = gridToLinearIndex(mod(xBox-1, nbBoxesX), mod(yBox-1, nbBoxesY), nbBoxesX)
    i7 = gridToLinearIndex(xBox, mod(yBox+1, nbBoxesY), nbBoxesX)
    i8 = gridToLinearIndex(xBox, mod(yBox-1, nbBoxesY), nbBoxesX)
    return (i1, i2, i3, i4, i5, i6, i7, i8, linearIndex)
end

"""
    updateNeighbours!(particleGrid::ParticleGrid2D, maxDist<:Real)

For each particle, find the other particles within a distance maxDist and add them to neighbourIndices.
"""
function updateNeighbours!(particleGrid::ParticleGrid2D, maxDist::Real)
    nbBoxesX = floor(Int64, (particleGrid.xmax - particleGrid.xmin)/maxDist)
    nbBoxesY = floor(Int64, (particleGrid.ymax - particleGrid.ymin)/maxDist)

    # Put particles into boxes
    updateVoxelInformation!(particleGrid::ParticleGrid2D, maxDist::Real)

    # Make set with all particles per box
    d = Dict{Int64, Vector{Int64}}()
    for (index, particle) in enumerate(particleGrid.grid)
        if particle.voxel in keys(d)
            push!(d[particle.voxel], index)
        else
            d[particle.voxel] = [index;]
        end
    end

    # Fill neighbourIndices
    for voxel in keys(d)
        tup = findNeighbouringVoxels(voxel, nbBoxesX, nbBoxesY)
        for particleIndex in d[voxel]
            particle = particleGrid.grid[particleIndex]
            empty!(particle.neighbourIndices)
            for nbVoxel in tup
                if haskey(d, nbVoxel)
                    for nbParticle in d[nbVoxel]
                        dist = getEuclideanDistance(particleGrid, particleIndex, nbParticle)
                        if (dist <= maxDist) && (particleIndex != nbParticle)
                            push!(particle.neighbourIndices, nbParticle)
                        end
                    end
                end
            end
            particle.A = Matrix{Float64}(undef, length(particle.neighbourIndices), 5)
            resize!(particle.alfaij, length(particle.neighbourIndices))
            resize!(particle.alfaijBar, length(particle.neighbourIndices))
            resize!(particle.betaij, length(particle.neighbourIndices))
            resize!(particle.betaijBar, length(particle.neighbourIndices))
            resize!(particle.gammaij, length(particle.neighbourIndices))
            resize!(particle.dxVec, length(particle.neighbourIndices))
            resize!(particle.dyVec, length(particle.neighbourIndices))
            resize!(particle.wVec, length(particle.neighbourIndices))
            resize!(particle.dfVec, length(particle.neighbourIndices))
        end
    end
end

"""
    findLocalExtrema!(particleGrid::ParticleGrid, particleIndex::Integer, fVec::Vector{Float64})::Tuple{Float64, Float64}

Find the minima and the maxima in the neighbourhood of particleIndex in fVec.
"""
function findLocalExtrema!(particleGrid::ParticleGrid, particleIndex::Integer, fVec::Vector{Float64})::Tuple{Float64, Float64}
    mini = fVec[particleIndex]
    maxi = fVec[particleIndex]
    for i in particleGrid.grid[particleIndex].neighbourIndices
        mini = min(mini, fVec[i])
        maxi = max(maxi, fVec[i])
    end
    return (mini, maxi)
end

function findLocalExtremaAbs!(particleGrid::ParticleGrid1D, particleIndex::Integer, fVec::Vector{Float64})::Tuple{Float64, Float64, Float64, Float64}
    mini = fVec[particleIndex]
    maxi = fVec[particleIndex]
    minAbs = abs(fVec[particleIndex])
    maxAbs = abs(fVec[particleIndex])
    for i in particleGrid.grid[particleIndex].neighbourIndices
        minAbs = min(abs(minAbs), abs(fVec[i]))
        maxAbs = max(abs(maxAbs), abs(fVec[i]))
        mini = min(mini, fVec[i])
        maxi = max(maxi, fVec[i])
    end
    return (mini, maxi, minAbs, maxAbs)
end

function findLocalExtremaAbs!(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Matrix{Float64})::Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}
    @assert size(fVec, 2) == 2
    mini1 = fVec[particleIndex, 1]
    maxi1 = fVec[particleIndex, 1]
    minAbs1 = abs(fVec[particleIndex, 1])
    maxAbs1 = abs(fVec[particleIndex, 1])
    mini2 = fVec[particleIndex, 2]
    maxi2 = fVec[particleIndex, 2]
    minAbs2 = abs(fVec[particleIndex, 2])
    maxAbs2 = abs(fVec[particleIndex, 2])
    for i in particleGrid.grid[particleIndex].neighbourIndices
        mini1 = min(mini1, fVec[i, 1])
        maxi1 = max(maxi1, fVec[i, 1])
        minAbs1 = min(abs(minAbs1), abs(fVec[i, 1]))
        maxAbs1 = max(abs(maxAbs1), abs(fVec[i, 1]))
        mini2 = min(mini2, fVec[i, 2])
        maxi2 = max(maxi2, fVec[i, 2])
        minAbs2 = min(abs(minAbs2), abs(fVec[i, 2]))
        maxAbs2 = max(abs(maxAbs2), abs(fVec[i, 2]))
    end
    return (mini1, maxi1, minAbs1, maxAbs1, mini2, maxi2, minAbs2, maxAbs2)
end

"""
    findLocalExtrema!(particleGrid::ParticleGrid, particleIndex::Integer, fMatrix::Matrix{Float64})::Tuple{Float64, Float64}

Find the minima and the maxima in the neighbourhood of particleIndex in every column of fMatrix.
"""
function findLocalExtrema!(particleGrid::ParticleGrid, particleIndex::Integer, fMatrix::Matrix{Float64})::Array{Float64}
    @assert size(fMatrix, 2) == 2  # Assume two columns
    minxx = fMatrix[particleIndex, 1]
    maxxx = fMatrix[particleIndex, 1]
    minyy = fMatrix[particleIndex, 2]
    maxyy = fMatrix[particleIndex, 2]
    for i in particleGrid.grid[particleIndex].neighbourIndices
        minxx = min(minxx, fMatrix[i, 1])
        maxxx = max(maxxx, fMatrix[i, 1])
        minyy = min(minyy, fMatrix[i, 2])
        maxyy = max(maxyy, fMatrix[i, 2])
    end
    return (minxx, maxxx, minyy, maxyy)
end


# -------------------- Saving and plotting routines --------------------
"""
    saveGrid(settings::SimSetting, particleGrid::ParticleGrid, time::Real)

Save particle grid to file at settings.saveDir*data/".
"""
function saveGrid(settings::SimSetting, particleGrid::ParticleGrid, time::Real)
    save("$(settings.saveDir)data/step$(settings.currentSaveNb).jld2", "time", time, "particleGrid", particleGrid)
    settings.currentSaveNb += 1
end

"""
    plotDensity(settings::SimSetting, timeIndex::Integer; saveFigure::Union{Bool, String}=false)::Plots.Plot

Plot the density at a time step, where timeIndex is the number of the savefile.
"""
function plotDensity(settings::SimSetting, timeIndex::Integer; saveFigure::Union{Bool, String}=false, ms::Integer=3, showMOODEvents::Bool=false)::Plots.Plot
    # Plot data
    d = load("$(settings.saveDir)data/step$(timeIndex).jld2")
    dataLabel = @sprintf "time: %.3E" d["time"]
    particleGrid = d["particleGrid"]
    rhos = map(particle -> particle.rho, particleGrid.grid)

    if particleGrid isa ParticleGrid1D
        min1 = minimum(rhos)
        max1 = maximum(rhos)    
        mass = sum((particle.rho*particle.volume for particle in particleGrid.grid))
        pos = map(particle -> particle.pos, particleGrid.grid)
        p = plot(pos, rhos, xlabel="x", ylabel=L"\rho", label=dataLabel)

        if showMOODEvents
            plot!(p, [pos[i] for i in eachindex(pos) if particleGrid.grid[i].moodEvent], [rhos[i] for i in eachindex(pos) if particleGrid.grid[i].moodEvent], label="", markershape=:star5, lt=:scatter)
        end
        # Add initial conditions
        d = load("$(settings.saveDir)data/step0.jld2")
        particleGrid = d["particleGrid"]
        rhos = map(particle -> particle.rho, particleGrid.grid)
        pos = map(particle -> particle.pos, particleGrid.grid)
        title = @sprintf("(min, max) = (%.3E, %.3E), mass = %.3E", min1, max1, mass)
        plot!(p, pos, rhos, label="Initial condition", seriescolor=:grey, linestyle=:dash, title=title)
    elseif particleGrid isa ParticleGrid2D
        min = minimum(rhos)
        max = maximum(rhos)
        posX = map(particle -> particle.pos[1], particleGrid.grid)
        posY = map(particle -> particle.pos[2], particleGrid.grid)
        mass = sum((particle.rho*particle.volume for particle in particleGrid.grid))
        title = @sprintf("Time: %.1f, (min, max) = (%.3E, %.3E), mass = %.3E", d["time"], min, max, mass)
        p = scatter(posX, posY, zcolor=rhos, size=(800, 800), ms=ms, colormap=:turbo, clim=(min, max), title=title)
    else
        error("ParticleGrid plotting function not implemented.")
    end

    # saving stuff
    if isa(saveFigure, String)
        savefig(p, saveFigure)
    elseif isa(saveFigure, Bool)
        if saveFigure 
            savefig(p, "$(settings.saveDir)figures/densityPlot$(timeIndex).png")
        end
    end
    return p
end

function animateDensity(settings::SimSetting; saveFigure::Union{Bool, String}=false, fps::Integer=10, ms::Integer=3, showMOODEvents::Bool=false)
    rhoAni = Animation()
    nbSaveSteps = length(filter(x->contains(x, "step"), readdir(settings.saveDir*"data")))
    particleGrid0 = load(settings.saveDir*"data/step0.jld2")["particleGrid"]
    m0 = sum((particle.rho*particle.volume for particle in particleGrid0.grid))
    titlefontsize = 10
    if particleGrid0 isa ParticleGrid1D
        for saveNb = 0:nbSaveSteps-1
            d = load(settings.saveDir*"data/step$(saveNb).jld2");
            particleGrid = d["particleGrid"];
            pos = map(particle -> particle.pos, particleGrid.grid);
            rho = map(particle -> particle.rho, particleGrid.grid);
            min1 = minimum(rho)
            max1 = maximum(rho)    
            mass = sum((particle.rho*particle.volume for particle in particleGrid.grid))
            title = @sprintf("Time: %.2f, extrema = (%.2E, %.2E), mass = %.2E", d["time"], min1, max1, mass/m0)
            rhoPlt = plot(pos, rho, xlabel="x", ylabel=L"\rho", legend=false, title=title, titlefontsize=titlefontsize)
            if showMOODEvents
                plot!(rhoPlt, [pos[i] for i in eachindex(pos) if particleGrid.grid[i].moodEvent], [rho[i] for i in eachindex(pos) if particleGrid.grid[i].moodEvent], label="", markershape=:star5, lt=:scatter)
            end
    
            frame(rhoAni, rhoPlt)
        end
    elseif particleGrid0 isa ParticleGrid2D
        # Get minimum and maximum value of plot
        min = minimum(particle -> particle.rho, particleGrid0.grid)
        max = maximum(particle -> particle.rho, particleGrid0.grid)
        for saveNb = 0:nbSaveSteps-1
            d = load(settings.saveDir*"data/step$(saveNb).jld2");
            particleGrid = d["particleGrid"];
            time = d["time"];
            posX = map(particle -> particle.pos[1], particleGrid.grid)
            posY = map(particle -> particle.pos[2], particleGrid.grid)
            rhos = map(particle -> particle.rho, particleGrid.grid)
            min1 = minimum(rhos)
            max1 = maximum(rhos)    
            mass = sum((particle.rho*particle.volume for particle in particleGrid.grid))
            title = @sprintf("Time: %.2f, extrema = (%.2E, %.2E), mass = %.2E", time, min1, max1, mass/m0)
            rhoPlt = scatter(posX, posY, zcolor=rhos, size=(800, 800), ms=ms, colormap=:turbo, clim=(min, max), title=title, titlefontsize=titlefontsize)
            frame(rhoAni, rhoPlt)
        end
    else
        error("ParticleGrid animation not implemented.")
    end

    # saving stuff
    if isa(saveFigure, String)
        gif(rhoAni, saveFigure, fps=fps)
    elseif isa(saveFigure, Bool)
        if saveFigure 
            gif(rhoAni, "$(settings.saveDir)figures/densityAnimation.gif", fps=fps)
        end
    end
end


end  # module ParticleGrids