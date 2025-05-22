module TimeIntegration

using LinearAlgebra
using IPlotPDESols
using ..ParticleGrids
using ..SimSettings
using ..ScalarHyperbolicEquations
using ..Interpolations

export mainTimeIntegrator!, mainTimeIntegrator2!

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

function (method::TimeStepper)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    error("Each `TimeStepper' must override the ()-operator.")
end

# Function called once before time integration loop to pre-calculate all relevant coefficients fot interpolation.
function initTimeStepper(method::TimeStepper, particleGrid::ParticleGrid, settings::SimSetting) end
function initTimeStepper(method::TimeStepper, particleGrids::Vector{T}, settings::SimSetting) where T <: ParticleGrid 
    @warn "Scalar timestepper given during initialization! Initializing each timestepper independently!"
    for particleGrid = particleGrids
        initTimeStepper(method, particleGrid, settings)
    end
end
function initTimeStepper(method::MeshfreeSystemTimeStepper, particleGrids::Vector{T}, settings::SimSetting) where T <: ParticleGrid
    @warn "Timestepper detected as a system timestepper, however, no initialization is used for this Timestepper."
end

include("FixedGridTimeSteppers.jl")
include("MeshfreeTimeSteppers.jl")
include("MeshfreeSystemTimeSteppers.jl")


"""
    mainTimeIntegrator!

This method performs that actual time integration. Provided with a timeStepper, equation, an initialized grid and simulation settings, it will perform
time integration.
"""
function mainTimeIntegrator!(timeStepper::TimeStepper, eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting)

    if !particleGrid.regular
        @assert timeStepper isa MeshfreeTimeStepper "Must use a MeshfreeTimeStepper for unstructured grids."
    end

    # Initialize grid
    updateNeighbours!(particleGrid, settings.interpRange)
    saveGrid(settings, particleGrid, 0.0)
    setCurvatures!(particleGrid, settings)

    # Initialize interpolation routine
    initTimeStepper(timeStepper, particleGrid, settings)
    t = 0.0
    k = 1
    time = @elapsed while t < settings.tmax
        dt = min(settings.dt, settings.tmax-t)

        # Save data every savefreq steps
        if mod(k, settings.saveFreq) == 0
            saveGrid(settings, particleGrid, t)
        end
        
        timeStepper(eq, particleGrid, settings, t, dt)

        t += dt
        k += 1
    end
    saveGrid(settings, particleGrid, t)
    saveSettings(settings)
    return time
end

"""
    mainTimeIntegrator!

This method performs that actual time integration. Provided with a timeStepper, equation, an initialized grid and simulation settings, it will perform
time integration.
This version uses the format required for the IPlotPDESols package. It will save the particleGrids as stats for compatibility issues.
Note that the usage differs from the function above: It does not save the grid! Instead the complete simulation data is returned.
This allows us to use the mainTimeIntegrator inside of the defining function for the simulationConfig!
"""
function mainTimeIntegrator2!(timeStepper::TimeStepper, eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting)
    
    if !particleGrid.regular
        @assert timeStepper isa MeshfreeTimeStepper "Must use a MeshfreeTimeStepper for unstructured grids."
    end

    # Initialize grid
    updateNeighbours!(particleGrid, settings.interpRange)

    # Initialize vectors for simulation data
    xs = [map(particle -> particle.pos, particleGrid.grid)]
    us = [map(particle -> particle.rho, particleGrid.grid)]
    ts = [0.0]
    #grids = [deepcopy(particleGrid.grid)]

    #saveGrid(settings, particleGrid, 0.0)
    setCurvatures!(particleGrid, settings)

    # Initialize interpolation routine
    initTimeStepper(timeStepper, particleGrid, settings)
    t = 0.0
    k = 1
    time = @elapsed while t < settings.tmax
        dt = min(settings.dt, settings.tmax-t)

        timeStepper(eq, particleGrid, settings, t, dt)
        t += dt

        # Save data every savefreq steps
        if mod(k, settings.saveFreq) == 0
            #saveGrid(settings, particleGrid, t)
            push!(xs, map(particle -> particle.pos, particleGrid.grid))
            push!(us, map(particle -> particle.rho, particleGrid.grid))
            push!(ts, t)
            #push!(grids, deepcopy(particleGrid.grid))
        end
        
        k += 1
    end
    #saveGrid(settings, particleGrid, t)
    push!(xs, map(particle -> particle.pos, particleGrid.grid))
    push!(us, map(particle -> particle.rho, particleGrid.grid))
    push!(ts, t)
    # push!(grids, deepcopy(particleGrid.grid))
    #saveSettings(settings)
    #sim_data = createSimData(xs, us, ts, params)#, ParamDict("saved_grids" => grids))
    return time, xs, us, ts
end


"""
    appendData!(xs_storage, us_storage, ts_storage, particle_grid::ParticleGrid1D, current_t::Real)

Appends data from a SCALAR `ParticleGrid1D` to storage vectors.
`us_storage` will store vectors of rho values (Vector{Float64}).
"""
function appendData!(
    xs_storage::Vector{Vector{T}} where T <: Union{Float64,Tuple}, 
    us_storage::Vector{Vector{Float64}}, # For scalar, this is Vector{Vector{Float64}}
    ts_storage::Vector{Float64}, 
    particle_grid::T where T <: ParticleGrid, 
    current_t::Real
)
    # Positions: Vector{Float64} for this time step
    current_xs = [p.pos for p in particle_grid.grid] # More efficient than map
    push!(xs_storage, current_xs)

    # Solution values: Vector{Float64} for this time step
    current_us = [p.rho for p in particle_grid.grid]
    push!(us_storage, current_us)
    
    push!(ts_storage, current_t)
end

"""
    appendData!(xs_storage, us_storage_sys, ts_storage, system_pg::Vector{ParticleGrid1D}, current_t::Real)

Appends data from a SYSTEM of `ParticleGrid1D` (represented as a Vector) to storage.
Assumes all component grids share the same particle positions and N_particles.
`us_storage_sys` will store a Vector where each element is a 
`Vector{Tuple{Vararg{Float64}}}` for that time step. Each tuple contains the
component values for a single particle.
"""
function appendData!(
    xs_storage::Vector{Vector{T}} where T <: Union{Float64,Tuple}, 
    us_storage_sys::Vector{Matrix{Float64}}, # e.g., Vector{Vector{Tuple{Float64, Float64}}}
    ts_storage::Vector{Float64}, 
    system_pg::Vector{<:ParticleGrid}, # Vector of ParticleGrid1D, one per component
    current_t::Real
)
    if isempty(system_pg)
        @warn "Attempting to append data from an empty system_pg."
        return
    end

    N_particles = length(system_pg[1].grid)
    N_components = length(system_pg)

    if N_particles == 0
        @warn "Particle grid for component 1 is empty."
        # Push empty position vector if desired, or handle error
        push!(xs_storage, Float64[])
        push!(us_storage_sys, Tuple{Vararg{Float64}}[]) # Pushes an empty Vector{Tuple{Vararg{Float64}}}
        push!(ts_storage, current_t)
        return
    end
    
    # Positions (from the first component grid, assumed consistent)
    current_xs = [p.pos for p in system_pg[1].grid]
    push!(xs_storage, current_xs)
    push!(ts_storage, current_t)

    current_step_us = Matrix{Float64}(undef, N_particles, N_components)

    for c_idx in 1:N_components
        current_step_us[:,c_idx] = [p.rho for p in system_pg[c_idx].grid]
    end
    push!(us_storage_sys, current_step_us)
end

function mainTimeIntegrator2!(
    system_timestepper::TimeStepper, 
    system_eq::Vector{T} where T <: ScalarHyperbolicEquation, # Your system equation type
    system_pg::Vector{T} where T <: ParticleGrid,
    settings::SimSetting 
)
    # ... (initial checks and updates for system_pg as before) ...

    xs_data = Vector{Vector{Float64}}()
    us_data_sys = Vector{Matrix{Float64}}() # Vector of (Vector of Tuples)

    for particleGrid in system_pg
        updateNeighbours!(particleGrid, settings.interpRange)
    end

    ts_data = Vector{Float64}()

    # Call the new appendData!
    appendData!(xs_data, us_data_sys, ts_data, system_pg, 0.0)

    # ... (initTimeStepper and setCurvatures for system as before) ...
    for scalar_grid_k in system_pg
        setCurvatures!(scalar_grid_k, settings) 
    end
    initTimeStepper(system_timestepper, system_pg, settings)

    t = 0.0
    k_step = 0
    elapsed_time = @elapsed while t < settings.tmax
        actual_dt = min(settings.dt, settings.tmax - t)

        system_timestepper(system_eq, system_pg, settings, t, actual_dt)
        t += actual_dt
        k_step += 1

        if mod(k_step, settings.saveFreq) == 0 || t >= settings.tmax
            appendData!(xs_data, us_data_sys, ts_data, system_pg, t)
        end
    end
    return elapsed_time, xs_data, us_data_sys, ts_data
end

end  # module TimeIntegration