module TimeIntegration

using LinearAlgebra
using ProgressMeter
using StaticArrays
using Base.Threads
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

"""
Ensures a vector `v` has at least capacity `n`.
Resizes if `length(v) < n`.
"""
function _ensure_capacity!(v::AbstractVector, n::Int)
    if length(v) < n
        n = n #+ n ÷ 4
        resize!(v, n)
    end
    return nothing
end


include("MeshfreeTimeSteppers.jl")
include("FixedGridTimeSteppers.jl")
include("MeshfreeSystemTimeSteppers.jl")

using StaticArrays

# --- Low-Level `saveData!` Helpers ---

"""
Helper for 1D positions (Vector{Float64}). Simply copies the data.
"""
function _copy_positions!(dest::Vector{Float64}, src::AbstractVector{Float64})
    copyto!(dest, src)
end

"""
Helper for 2D positions (Vector{SVector} -> Vector{Tuple}).
Converts SVectors to Tuples for plotting compatibility while copying.
"""
function _copy_positions!(dest::Vector{Tuple{Float64,Float64}}, src::AbstractVector{SVector{2, Float64}})
    @inbounds for i in eachindex(dest, src)
        dest[i] = Tuple(src[i])
    end
end

"""
Saves data from a SCALAR particle grid into pre-allocated storage slots.
"""
function saveData!(xs_storage, us_storage, ts_storage, snap_idx::Int, 
                   particle_grid, current_t, remove_ghosts::Bool)
    
    # Save the current time
    ts_storage[snap_idx] = current_t
    
    # Get the destination vectors for this snapshot
    dest_pos = xs_storage[snap_idx]
    dest_rho = us_storage[snap_idx]
    
    if remove_ghosts
        # Get logical indices of non-ghost particles
        indices = .!particle_grid.is_boundary
        
        # Use views to copy only the non-ghost data
        _copy_positions!(dest_pos, view(particle_grid.positions, indices))
        copyto!(dest_rho, view(particle_grid.rhos, indices))
    else
        # Copy all data
        _copy_positions!(dest_pos, particle_grid.positions)
        copyto!(dest_rho, particle_grid.rhos)
    end
end

"""
Saves data from a SYSTEM of particle grids into pre-allocated storage.
"""
function saveData!(xs_storage, us_storage, ts_storage, snap_idx::Int, 
                   system_pgs::Tuple, current_t, remove_ghosts::Bool)
    
    # Save the current time
    ts_storage[snap_idx] = current_t
    
    first_pg = system_pgs[1]
    dest_pos = xs_storage[snap_idx]
    dest_rho_matrix = us_storage[snap_idx] # This is a Matrix
    
    if remove_ghosts
        # Get logical indices of non-ghost particles from the first grid
        indices = .!first_pg.is_boundary
        
        # Copy positions from the first grid
        _copy_positions!(dest_pos, view(first_pg.positions, indices))
        
        # Copy rhos from each grid as a column in the destination matrix
        for (k, pg) in enumerate(system_pgs)
            copyto!(view(dest_rho_matrix, :, k), view(pg.rhos, indices))
        end
    else
        # Copy all positions
        _copy_positions!(dest_pos, first_pg.positions)
        
        # Copy all rhos
        for (k, pg) in enumerate(system_pgs)
            copyto!(view(dest_rho_matrix, :, k), pg.rhos)
        end
    end
end


"""
Optimized main time integrator for a SCALAR equation.
"""
function mainTimeIntegrator!(
    timeStepper::TimeStepper, 
    eq::ScalarHyperbolicPDE{D}, 
    particleGrid::ParticleGrid, 
    settings::SimSetting;
    snapshots::Integer = 10, 
    remove_ghosts::Bool = false
) where {D}
    # --- Initialization ---
    initTimeStepper(timeStepper, particleGrid, settings)
    
    # --- Pre-allocate Storage ---
    # Determine the number of particles to save
    N_save = remove_ghosts ? (particleGrid.N - particleGrid.N_ghost) : particleGrid.N
    
    # Define the position type (Float64 for 1D, Tuple for 2D)
    pos_type = D == 1 ? Float64 : Tuple{Float64,Float64}
    
    # Pre-allocate the storage vectors
    xs = [Vector{pos_type}(undef, N_save) for _ in 1:snapshots]
    us = [Vector{Float64}(undef, N_save) for _ in 1:snapshots]
    ts = Vector{Float64}(undef, snapshots)

    # --- Snapshot Time Points ---
    # Create an array of equidistant time points, including t=0 and t=tmax
    t_snap = range(0.0, settings.tmax, length=snapshots)
    snap_counter = 1

    # Save initial state (t=0)
    saveData!(xs, us, ts, snap_counter, particleGrid, t_snap[snap_counter], remove_ghosts)
    snap_counter += 1 # We are now looking for the 2nd snapshot

    # --- Main Time Loop ---
    t = 0.0
    k_step = 0
    p = Progress(convert(Int, ceil(settings.tmax / settings.dt)), "Running Scalar Simulation...")
    
    elapsed_time = @elapsed while t < settings.tmax && snap_counter <= snapshots
        actual_dt = min(settings.dt, settings.tmax - t)
        if actual_dt <= 1e-12; break; end

        timeStepper(eq, particleGrid, settings, t, actual_dt)
        
        t += actual_dt
        k_step += 1

        # Check for and save snapshots
        # Use a while-loop in case dt spans multiple snapshot times
        while snap_counter <= snapshots && t >= t_snap[snap_counter]
            # Save data at the *exact* snapshot time
            saveData!(xs, us, ts, snap_counter, particleGrid, t_snap[snap_counter], remove_ghosts)
            snap_counter += 1
        end
        
        ProgressMeter.next!(p)
    end
    num_saved_snapshots = snap_counter - 1
    return elapsed_time, xs[1:num_saved_snapshots], us[1:num_saved_snapshots], ts[1:num_saved_snapshots]
end


"""
Optimized main time integrator for a SYSTEM of equations, using Tuples for performance.
"""
function mainTimeIntegrator!(
    system_timestepper::TimeStepper, 
    system_eqs::DiagonalHyperbolicSystem{N,D},
    system_pgs::ParticleGridSystem{N},
    settings::SimSetting;
    snapshots::Integer,
    remove_ghosts::Bool = false
) where {N,D}
    # --- Initialization ---
    for pg in system_pgs
        #updateNeighbors!(pg, settings.interpRange)
    end
    initTimeStepper(system_timestepper, system_pgs, settings)

    # --- Pre-allocate Storage ---
    first_pg = system_pgs[1]
    N_save = remove_ghosts ? (first_pg.N - first_pg.N_ghost) : first_pg.N
    N_vars = N # Number of variables in the system
    
    pos_type = D == 1 ? Float64 : Tuple{Float64,Float64}
    
    # Pre-allocate storage
    xs = [Vector{pos_type}(undef, N_save) for _ in 1:snapshots]
    us_sys = [Matrix{Float64}(undef, N_save, N_vars) for _ in 1:snapshots]
    ts = Vector{Float64}(undef, snapshots)

    # --- Snapshot Time Points ---
    t_snap = range(0.0, settings.tmax, length=snapshots)
    snap_counter = 1
    
    # Save initial state (t=0)
    saveData!(xs, us_sys, ts, snap_counter, system_pgs, t, remove_ghosts)
    snap_counter += 1 # We are now looking for the 2nd snapshot

    # --- Main Time Loop ---
    t = 0.0
    k_step = 0
    p = Progress(convert(Int, ceil(settings.tmax / settings.dt)), "Running System Simulation...")

    elapsed_time = @elapsed while t < settings.tmax && snap_counter <= snapshots
        actual_dt = min(settings.dt, settings.tmax - t)
        if actual_dt <= 1e-12; break; end

        for pg in system_pgs
            apply_boundary_conditions!(pg)
        end
        system_timestepper(system_eqs, system_pgs, settings, t, actual_dt)
        
        t += actual_dt
        k_step += 1

        # Check for and save snapshots
        # Use a while-loop in case dt spans multiple snapshot times
        while snap_counter <= snapshots && t >= t_snap[snap_counter]
            # Save data at the *exact* snapshot time
            saveData!(xs, us_sys, ts, snap_counter, system_pgs, t, remove_ghosts)
            snap_counter += 1
        end
        
        ProgressMeter.next!(p)
    end
    num_saved_snapshots = snap_counter - 1
    return elapsed_time, xs[1:num_saved_snapshots], us[1:num_saved_snapshots], ts[1:num_saved_snapshots]
end

end  # module TimeIntegration