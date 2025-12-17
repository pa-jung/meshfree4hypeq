module GridMovement

using ..ParticleGrids
using ..HyperbolicPDEs

export GridMover, NoGridMover, CustomGridMover, PhysicalGridMover

abstract type GridMover end

struct NoGridMover <: GridMover end

struct CustomGridMover <: GridMover
    vel_func::Function
    params::Tuple
end

struct PhysicalGridMover{E} <: GridMover
    pde::E
end

# Default function, no grid move
function (gm::NoGridMover)(pg::ParticleGrid, dt::Real); return; end

function (gm::CustomGridMover)(pg::ParticleGrid, dt::Real); 
    positions = pg.positions
    rhos = pg.rhos
    vel_func = gm.vel_func
    for p_idx = 1:pg.N
        rho = rhos[p_idx]
        positions[p_idx] += vel_func(positons[p_idx],rho,gm.params) * dt
    end
    updateNeighbors!(pg)
    manage_particles!(pg)
    sort_1d_particles!(pg)
    updateNeighbors!(pg)    
    return
end 

function (gm::PhysicalGridMover{BurgersEquation{a}})(pg::ParticleGrid1D, dt::Real) where {a}
    rhos = pg.rhos
    positions = pg.positions
    for p_idx = 1:pg.N
        positions[p_idx] += a * rhos[p_idx] * dt
    end
    #sort_1d_particles!(pg)
    updateNeighbors!(pg)
    manage_particles!(pg)  
    return    
end

function moveGrid!(::BurgersEquation{a}, pg::ParticleGrid1D, dt::Float64) where {a}
    positions = pg.positions
    rhos = pg.rhos

    for (p_idx, rho) = enumerate(rhos)
        positions[p_idx] += a * 1/2 * dt
    end
    sort_1d_particles!(pg)
    updateNeighbors!(pg)
    return
end

function moveGrid!(::LinearAdvection{a}, pg::ParticleGrid1D, dt::Float64) where {a}
    positions = pg.positions
    pg.xmax += a * dt
    pg.xmin += a * dt
    for p_idx = eachindex(positions)
        positions[p_idx] += a * dt
    end
    updateNeighbors!(pg)
    return
end


function moveGrid!(::BurgersEquation{0.0}, pg::ParticleGrid1D, dt::Float64)
    return
end

end