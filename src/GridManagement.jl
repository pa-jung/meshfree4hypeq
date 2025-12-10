module GridManagement

using ..ParticleGrids

abstract type GridManager end

struct GridManager1D <: GridManager
    # Parameters for grid management 
    radius::Float64
    nb_min::Int

    # Buffers
    merged::BitVector

    function GridManager1D(radius::Float64, nb_min::Int)
        new(radius, nb_min, falses(0))
    end
end



# export GridMover, NoGridMover, CustomGridMover, PhysicalGridMover

# abstract type GridMover end

# struct NoGridMover <: GridMover end

# struct CustomGridMover <: GridMover
#     vel_func::Function
#     params::Tuple
# end

# struct PhysicalGridMover{E,IC} <: GridMover
#     pde::E
#     ic::IC
# end

# # Default function, no grid move
# function (gm::NoGridMover)(pg::ParticleGrid, dt::Real); return; end

# function (gm::CustomGridMover)(pg::ParticleGrid, dt::Real); 
#     positions = pg.positions
#     rhos = pg.rhos
#     vel_func = gm.vel_func
#     for (p_idx, pos) = enumerate(positions)
#         rho = rhos[p_idx]
#         positions[p_idx] += vel_func(pos,rho,gm.params) * dt
#     end
#     sort_1d_particles!(pg)
#     updateNeighbors!(pg)    
#     return
# end 

# function (gm::PhysicalGridMover{BurgersEquation{a},Riemann{Float64,Float64}})(pg::ParticleGrid1D, dt::Real) where {a}
#     uL = gm.ic.uL
#     uR = gm.ic.uR
    
#     if (pg.bc == :periodic) || (uL < uR)
#         vel = max(0,uL)
#     else
#         vel = 0.5 * (uL + uR)
#     end
#     positions = pg.positions
#     for p_idx = eachindex(positions)
#         positions[p_idx] += a * vel * dt
#     end
#     sort_1d_particles!(pg)
#     updateNeighbors!(pg)
#     return    
# end

# function moveGrid!(::BurgersEquation{a}, pg::ParticleGrid1D, dt::Float64) where {a}
#     positions = pg.positions
#     rhos = pg.rhos

#     for (p_idx, rho) = enumerate(rhos)
#         positions[p_idx] += a * 1/2 * dt
#     end
#     sort_1d_particles!(pg)
#     updateNeighbors!(pg)
#     return
# end

# function moveGrid!(::LinearAdvection{a}, pg::ParticleGrid1D, dt::Float64) where {a}
#     positions = pg.positions
#     pg.xmax += a * dt
#     pg.xmin += a * dt
#     for p_idx = eachindex(positions)
#         positions[p_idx] += a * dt
#     end
#     updateNeighbors!(pg)
#     return
# end


# function moveGrid!(::BurgersEquation{0.0}, pg::ParticleGrid1D, dt::Float64)
#     return
# end

end