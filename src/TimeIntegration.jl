module TimeIntegration

using LinearAlgebra
using ProgressMeter
using ..ParticleGrids
using ..SimSettings
using ..HyperbolicPDEs
using ..Interpolations
using ..SourceTerms
using ..ImplicitSolvers

export mainTimeIntegrator!, mainTimeIntegratorNew!

"""
    TimeStepper

Time integration methods are split into two groups: MeshfreeTimeStepper and FixedGridTimeStepper, the former working for all types of grids, 
the latter only working for fixed grid with uniform discretisation parameter. Time integration methods are implemented as structs that override
the ()-operator. See EulerUpwind in "MeshfreeTimeSteppers.jl" as an example. 
"""
abstract type TimeStepper end
abstract type MeshfreeTimeStepper <: TimeStepper end
abstract type FixedGridTimeStepper <: TimeStepper end
abstract type MeshfreeSystemTimeStepper <: MeshfreeTimeStepper end

function (method::TimeStepper)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    error("Each `TimeStepper' must override the ()-operator.")
end

# Function called once before time integration loop to pre-calculate all relevant coefficients fot interpolation.
function initTimeStepper(method::FixedGridTimeStepper, particleGrid::ParticleGrid, settings::SimSetting)
    @info "Simulation uses $(typeof(method)) on a fixed regular Grid!"
end
function initTimeStepper(method::MeshfreeTimeStepper, particleGrid::ParticleGrid, settings::SimSetting) 
    fallback_string = "none"
    if hasfield(typeof(method), :fallbackInterpolator) & !isnothing(method.fallbackInterpolator)
        fallback_string = "$(typeof(method.fallbackInterpolator))"
    end
    @info "Simulation uses $(typeof(method)) with main gradient: $(typeof(method.gradientInterpolator)) and fallback gradient: $fallback_string"
end
function initTimeStepper(method::TimeStepper, particleGrids::Vector{T}, settings::SimSetting) where T <: ParticleGrid 
    @warn "Scalar timestepper given during initialization! Initializing each timestepper independently!"
    for particleGrid = particleGrids
        initTimeStepper(method, particleGrid, settings)
    end
end
function initTimeStepper(method::MeshfreeSystemTimeStepper, particleGrids::Vector{T}, settings::SimSetting) where T <: ParticleGrid
    @warn "Timestepper detected as a system timestepper, however, no initialization is used for this Timestepper."
end

include("MeshfreeTimeSteppers.jl")
include("FixedGridTimeSteppers.jl")
include("MeshfreeSystemTimeSteppers.jl")


"""
Appends data from a SCALAR SoA ParticleGrid to storage vectors.
"""
function appendData!(
    xs_storage::Vector, 
    us_storage::Vector{<:AbstractVector{Float64}},
    ts_storage::Vector{Float64}, 
    particle_grid::ParticleGrid, 
    current_t::Real
)
    interior = particle_grid.interior_indices
    # Directly copy the interior data from the SoA vectors, which is much
    # more efficient than iterating through a vector of structs.
    push!(xs_storage, particle_grid.positions[interior])
    push!(us_storage, particle_grid.rhos[interior])
    push!(ts_storage, current_t)
end

"""
Appends data from a SYSTEM of SoA particle grids (passed as a Tuple).
The solution `u` for a single time step is stored as a single matrix (particles x components).
"""
function appendData!(
    xs_storage::Vector, 
    us_storage::Vector{<:AbstractMatrix{Float64}},
    ts_storage::Vector{Float64}, 
    system_pgs::Tuple{Vararg{<:ParticleGrid}},
    current_t::Real
)
    # Positions are taken from the first grid, assuming they are all identical
    first_grid = system_pgs[1]
    interior = first_grid.interior_indices
    push!(xs_storage, first_grid.positions[interior])

    # --- SIMPLIFIED: Build the matrix of `rho` values efficiently ---
    # 1. Create a generator that yields the interior rho vector for each component grid.
    rho_vectors = (view(pg.rhos, interior) for pg in system_pgs)
    # 2. Use `hcat` to efficiently concatenate these column vectors into a single matrix.
    push!(us_storage, hcat(rho_vectors...))
    
    push!(ts_storage, current_t)
end

"""
Optimized main time integrator for a SCALAR equation.
"""
function mainTimeIntegrator!(
    timeStepper::TimeStepper, 
    eq::ScalarHyperbolicPDE, 
    particleGrid::ParticleGrid, 
    settings::SimSetting
)
    # --- Initialization ---
    updateNeighbours!(particleGrid, settings.interpRange)
    initTimeStepper(timeStepper, particleGrid, settings)
    
    # Initialize storage with the correct types for a scalar simulation
    pos_type = typeof(particleGrid.positions[1])
    xs = Vector{Vector{pos_type}}()
    us = Vector{Vector{Float64}}()
    ts = Vector{Float64}()

    appendData!(xs, us, ts, particleGrid, 0.0)

    # --- Main Time Loop ---
    t = 0.0
    k_step = 0
    p = Progress(convert(Int, ceil(settings.tmax / settings.dt)), "Running Scalar Simulation...")
    
    elapsed_time = @elapsed while t < settings.tmax
        actual_dt = min(settings.dt, settings.tmax - t)
        if actual_dt <= 1e-12; break; end

        apply_boundary_conditions!(particleGrid)
        timeStepper(eq, particleGrid, settings, t, actual_dt)
        
        t += actual_dt
        k_step += 1

        if mod(k_step, settings.saveFreq) == 0 || t >= settings.tmax
            appendData!(xs, us, ts, particleGrid, t)
        end
        ProgressMeter.next!(p)
    end
    
    return elapsed_time, xs, us, ts
end


"""
Optimized main time integrator for a SYSTEM of equations, using Tuples for performance.
"""
function mainTimeIntegrator!(
    system_timestepper::TimeStepper, 
    system_eqs::DiagonalHyperbolicSystem{N,D},
    system_pgs::ParticleGridSystem{N},
    settings::SimSetting 
) where {N,D}
    # --- Initialization ---
    for pg in system_pgs
        updateNeighbours!(pg, settings.interpRange)
    end
    initTimeStepper(system_timestepper, system_pgs, settings)

    # Initialize storage with the correct types for a system simulation
    pos_type = typeof(system_pgs[1].positions[1])
    xs = Vector{Vector{pos_type}}()
    us_sys = Vector{Matrix{Float64}}() # Storing each time step as a Matrix
    ts = Vector{Float64}()
    
    appendData!(xs, us_sys, ts, system_pgs, 0.0)

    # --- Main Time Loop ---
    t = 0.0
    k_step = 0
    p = Progress(convert(Int, ceil(settings.tmax / settings.dt)), "Running System Simulation...")

    elapsed_time = @elapsed while t < settings.tmax
        actual_dt = min(settings.dt, settings.tmax - t)
        if actual_dt <= 1e-12; break; end

        for pg in system_pgs
            apply_boundary_conditions!(pg)
        end
        system_timestepper(system_eqs, system_pgs, settings, t, actual_dt)
        
        t += actual_dt
        k_step += 1

        if mod(k_step, settings.saveFreq) == 0 || t >= settings.tmax
            appendData!(xs, us_sys, ts, system_pgs, t)
        end
        ProgressMeter.next!(p)
    end

    return elapsed_time, xs, us_sys, ts
end

end  # module TimeIntegration