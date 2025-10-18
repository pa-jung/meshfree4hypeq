module TimeIntegration

using LinearAlgebra
using ProgressMeter
using StaticArrays
using ..ParticleGrids
using ..SimSettings
using ..HyperbolicPDEs
using ..Interpolations
using ..SourceTerms
using ..ImplicitSolvers
using ..MOOD
using ..MLSWeightFunctions

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

using StaticArrays

# --- Low-Level `appendData!` Helpers ---

# Fallback method: For simple types (like a single vector or a number), just push it.
appendData!(storage::Vector, data) = push!(storage, data)

"""
Specialized method for `Vector{SVector}`.
Converts SVectors to Tuples for plotting compatibility before pushing.
"""
function appendData!(storage::Vector, data::AbstractVector{<:SVector})
    # Create a new vector of Tuples from the SVectors
    data_as_tuples = [Tuple(p) for p in data]
    push!(storage, data_as_tuples)
end

"""
Specialized method for the generator used in the system case.
Collects the vectors into a matrix using `hcat`.
"""
function appendData!(storage::Vector, data::Base.Generator)
    # The `...` splats the generator contents into the hcat function
    push!(storage, hcat(data...))
end

# --- High-Level `appendData!` Wrappers ---

"""
Appends data from a SCALAR particle grid. Dispatches to the correct
low-level helper for each data type.
"""
function appendData!(xs_storage, us_storage, ts_storage, particle_grid, current_t)
    interior = particle_grid.interior_indices
    
    # Extract the data (using views to avoid allocations)
    pos_data = @view particle_grid.positions[interior]
    rho_data = @view particle_grid.rhos[interior]
    
    # Let multiple dispatch choose the correct helper for each type
    appendData!(xs_storage, pos_data)
    appendData!(us_storage, rho_data)
    appendData!(ts_storage, current_t)
end

"""
Appends data from a SYSTEM of particle grids. Dispatches to the correct
low-level helper for each data type.
"""
function appendData!(xs_storage, us_storage, ts_storage, system_pgs::Tuple, current_t)
    interior = system_pgs[1].interior_indices
    
    # Extract the position data
    pos_data = @view system_pgs[1].positions[interior]
    
    # Create the generator for the rho data
    rho_generator = (view(pg.rhos, interior) for pg in system_pgs)
    
    # Let multiple dispatch choose the correct helper for each type
    appendData!(xs_storage, pos_data)
    appendData!(us_storage, rho_generator)
    appendData!(ts_storage, current_t)
end

"""
Optimized main time integrator for a SCALAR equation.
"""
function mainTimeIntegrator!(
    timeStepper::TimeStepper, 
    eq::ScalarHyperbolicPDE{D}, 
    particleGrid::ParticleGrid, 
    settings::SimSetting
) where {D}
    # --- Initialization ---
    #updateNeighbors!(particleGrid, settings.interpRange)
    initTimeStepper(timeStepper, particleGrid, settings)
    
    # Initialize storage with the correct types for a scalar simulation
    pos_type = D == 1 ? Float64 : Tuple{Float64,Float64}
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
        #updateNeighbors!(pg, settings.interpRange)
    end
    initTimeStepper(system_timestepper, system_pgs, settings)

    # Initialize storage with the correct types for a system simulation
    pos_type = D == 1 ? Float64 : Tuple{Float64,Float64}
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