export EulerUpwind, Upwind, RalstonRK2, RK3, RK4, RalstonRK2Limiter, RalstonRK2SmoothSwitch, RalstonRK2SmoothSwitch2

using Base.Threads
# This assumes your ParticleGrid abstract type is accessible, e.g., via:
# using ..ParticleGrids 

# This function assumes that any timestepper `ts` you pass to it will have 
# mutable fields `neighbor_fs::Vector{Float64}` and `neighbor_dfs::Vector{Float64}`.


"""
    check_for_nans(s::Any; range=nothing, counter=nothing)

Inspects all `AbstractVector{<:AbstractFloat}` fields within a given struct `s` for `NaN` values.

If NaNs are found, it throws a detailed error specifying which fields contain NaNs,
how many there are, and a list of the indices where they were found.

# Arguments
- `s::Any`: The struct to inspect.
- `range::Union{UnitRange, Nothing}=nothing`: An optional range of indices to check within each vector. If `nothing`, the entire vector is checked.
- `counter::Union{Base.Threads.Atomic{Int}, Nothing}=nothing`: An optional atomic counter to track calls.

# Example
```julia
debug_counter = Threads.Atomic{Int}(0)
# ... inside a loop
check_for_nans(my_struct, counter=debug_counter)
```
"""
function check_for_nans(s::Any; range::Union{UnitRange, Nothing}=nothing, counter::Union{Base.Threads.Atomic{Int}, Nothing}=nothing)
    # --- NEW: Atomic Counter Logic ---
    # If a counter is provided, increment it and optionally print a debug message.
    if !isnothing(counter)
        Threads.atomic_add!(counter, 1)
        current_count = counter[]

        # Example: Print a message every 1000 calls.
        if mod(current_count, 1000) == 0
            println("Running NaN check #$(current_count)...")
        end
    end

    # A dictionary to store the field name and a list of indices where NaNs are found.
    nan_locations = Dict{Symbol, Vector{Int}}()

    # Iterate over all property names (fields) of the struct.
    for field_name in propertynames(s)
        field_value = getproperty(s, field_name)

        # We only care about vectors of floating-point numbers.
        if !(field_value isa AbstractVector{<:AbstractFloat})
            continue
        end

        # Determine the actual range to iterate over.
        check_range = isnothing(range) ? eachindex(field_value) : range
        
        # Safety check to prevent BoundsError if the provided range is too large.
        if last(check_range) > length(field_value)
            println("Warning: Skipping field `:$field_name` in check_for_nans because the provided range is out of bounds.")
            continue
        end

        # Iterate and check for NaNs.
        for i in check_range
            if isnan(field_value[i])
                if !haskey(nan_locations, field_name)
                    nan_locations[field_name] = Int[]
                end
                push!(nan_locations[field_name], i)
            end
        end
    end

    # If the dictionary is not empty, it means we found NaNs.
    if !isempty(nan_locations)
        error_message = "NaNs detected!\n"
        if !isnothing(counter)
            error_message *= "(Check count: $(counter[]))\n"
        end
        
        # Build a detailed error message.
        for (field, indices) in nan_locations
            count = length(indices)
            error_message *= "Field `:$field`: Found $count NaN(s) at indices:\n"
            
            max_indices_to_show = 20
            indices_str = join(indices[1:min(count, max_indices_to_show)], ", ")
            if count > max_indices_to_show
                indices_str *= ", ..."
            end
            error_message *= "  [$indices_str]\n"
        end
        
        error(error_message)
    end

    return nothing
end

"""
Ensures a vector `v` has at least capacity `n`.
Resizes if `length(v) < n`.
"""
function _ensure_capacity!(v::Vector, n::Int)
    if length(v) < n
        n = n + n ÷ 4
        resize!(v, n)
    end
    return nothing
end

"""
    initTS!(ts::MeshfreeTimeStepper, pg::ParticleGrid, fVec::AbstractVector)

Initializes all buffers within the timestepper `ts` based on the particle grid `pg`.
It then runs a parallel "pre-gather" loop to fill the `neighbor_fs` and 
`neighbor_dfs` buffers using data from `fVec`.
"""
function initTS!(
    ts::T,
    pg, # Keep it generic, just need to access its fields
    fVec::AbstractVector{<:Real}
) where {T <: MeshfreeTimeStepper}

    # --- 1. Get Required Buffer Sizes ---
    # `num_particles` is the number of particles (N)
    num_particles = length(pg.num_neighbors) 
    # `num_interactions` is the total length of the flat neighbor lists (M)
    num_interactions = length(pg.neighbor_indices) 

    # --- 2. Resize Per-Particle Buffers (Size N) ---
    _ensure_capacity!(ts.rhoInit, num_particles)
    _ensure_capacity!(ts.rhos, num_particles)
    _ensure_capacity!(ts.div1, num_particles)
    
    # --- 3. Resize Per-Interaction Buffers (Size M) ---
    _ensure_capacity!(ts.neighbor_fs, num_interactions)
    _ensure_capacity!(ts.neighbor_dfs, num_interactions)

    # --- 4. Parallel Pre-Gather Loop ---
    # Get local aliases to the buffers for cleaner code in the loop
    neighbor_fs  = ts.neighbor_fs
    neighbor_dfs = ts.neighbor_dfs

    # This is a "pleasantly parallel" loop. Each thread `i` writes to
    # a unique, non-overlapping slice of `neighbor_fs` and `neighbor_dfs`,
    # so there are no race conditions.
    Threads.@threads for i in 1:num_particles
        f_i = fVec[i]
        num_nb = pg.num_neighbors[i]

        if num_nb == 0
            continue
        end

        # Get the unique slice of indices for this particle's neighbors
        start_idx = pg.neighbor_pointers[i]
        neighbor_slice = start_idx:(start_idx + num_nb - 1)

        @inbounds for k in neighbor_slice
            # `k` is the global index into the flat neighbor arrays
            
            # 1. GATHER: Get neighbor index `j`...
            j = pg.neighbor_indices[k]
            # ...and then get its value `f_j`. This is the slow part.
            f_j = fVec[j] 

            # 2. CALCULATE & STORE: Write to the pre-allocated buffers
            neighbor_fs[k]  = f_j
            neighbor_dfs[k] = f_j - f_i
        end
    end
    
    return nothing
end


# A simple example for EulerUpwind
struct EulerUpwind{G <: GradientInterpolator} <: MeshfreeTimeStepper
    gradientInterpolator::G
    rho_buffer::Vector{Float64}

    function EulerUpwind(gradientInterpolator::G) where {G <: GradientInterpolator}
        new{G}(gradientInterpolator, Float64[])
    end
end

function (eu::EulerUpwind)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    if N > length(eu.rho_buffer); resize!(eu.rho_buffer, N) end
    eu.rho_buffer[1:N] .= particleGrid.rhos # Store initial state for the step
    
    initTimeStep(eu.gradientInterpolator, particleGrid)

    for p_idx in 1:N
        if particleGrid.is_boundary[p_idx]; return; end
        div = eu.gradientInterpolator(particleGrid, p_idx, eu.rho_buffer, eq, settings)
        particleGrid.rhos[p_idx] = eu.rho_buffer[p_idx] - dt * div
    end
end

function initTimeStepper(euler::EulerUpwind, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(euler.gradientInterpolator, particleGrid)
end

struct RK3{G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}      # Stores the solution at the start of the step
    rho_stage1::Vector{Float64} # Stores the result of the first stage
    rho_stage2::Vector{Float64} # Stores the result of the second stage
    
    div1::Vector{Float64} # Stores divergence from stage 1
    div2::Vector{Float64} # Stores divergence from stage 2
    div3::Vector{Float64} # Stores divergence from stage 3

    function RK3(grad::G1, fallback::G2, mood::M) where {G1, G2, M}
        # Initialize with empty buffers; they will be resized on the first call
        new{G1, G2, M}(grad, fallback, mood, Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

# --- User-Friendly Constructor ---
function RK3(gradientInterpolator::G1; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1, G2, M}
    RK3(gradientInterpolator, fallbackInterpolator, mood)
end

function initTimeStepper(rk3::RK3, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !(rk3.fallbackInterpolator isa NoFallbackGrad)
        initTimeStep(rk3.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
end

function (rk3::RK3)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    #initMOOD!(rk3.mood,particleGrid.max_volume)
    # --- Ensure buffers are correctly sized for the current grid ---
    if length(rk3.rho_n) != N
        resize!.((rk3.rho_n, rk3.rho_stage1, rk3.rho_stage2, rk3.div1, rk3.div2, rk3.div3), N)
    end

    interior = particleGrid.interior_indices
    rk3.rho_n .= particleGrid.rhos # Store u^n

    # --- Stage 1: u^(1) = u^n + dt * L(u^n) ---
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !(rk3.fallbackInterpolator isa NoFallbackGrad); initTimeStep(rk3.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange); end

    for p_idx in interior
        # L(u^n) is -div(u^n)
        rk3.div1[p_idx] = rk3.gradientInterpolator(particleGrid, p_idx, rk3.rho_n, eq, settings)
        
        # Calculate candidate for stage 1
        rho_candidate = rk3.rho_n[p_idx] - dt * rk3.div1[p_idx]
        
        # Apply MOOD if necessary
        if !(rk3.fallbackInterpolator isa NoFallbackGrad) && rk3.mood(particleGrid, p_idx, rk3.rho_n, rho_candidate; firstStage=true)
            rk3.div1[p_idx] = rk3.fallbackInterpolator(particleGrid, p_idx, rk3.rho_n, eq, settings; setCurvature=false)
            rho_candidate = rk3.rho_n[p_idx] - dt * rk3.div1[p_idx]
        end
        rk3.rho_stage1[p_idx] = rho_candidate
    end
    particleGrid.rhos[interior] .= @view rk3.rho_stage1[interior]
    apply_boundary_conditions!(particleGrid) # CRITICAL: Update ghost cells for next stage
    rk3.rho_stage1 .= particleGrid.rhos # Update buffer with correct ghost cells

    # --- Stage 2: u^(2) = 3/4 u^n + 1/4 u^(1) + 1/4 dt * L(u^(1)) ---
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    
    for p_idx in interior
        rk3.div2[p_idx] = rk3.gradientInterpolator(particleGrid, p_idx, rk3.rho_stage1, eq, settings)
        
        rho_candidate = 0.75 * rk3.rho_n[p_idx] + 0.25 * rk3.rho_stage1[p_idx] - 0.25 * dt * rk3.div2[p_idx]
        
        if !(rk3.fallbackInterpolator isa NoFallbackGrad) && rk3.mood(particleGrid, p_idx, rk3.rho_stage1, rho_candidate)
            rk3.div2[p_idx] = rk3.fallbackInterpolator(particleGrid, p_idx, rk3.rho_stage1, eq, settings; setCurvature=false)
            rho_candidate = 0.75 * rk3.rho_n[p_idx] + 0.25 * rk3.rho_stage1[p_idx] - 0.25 * dt * rk3.div2[p_idx]
        end
        rk3.rho_stage2[p_idx] = rho_candidate
    end
    particleGrid.rhos[interior] .= @view rk3.rho_stage2[interior]
    apply_boundary_conditions!(particleGrid) # CRITICAL: Update ghost cells for next stage
    rk3.rho_stage2 .= particleGrid.rhos

    # --- Stage 3 (Final): u^{n+1} = 1/3 u^n + 2/3 u^(2) + 2/3 dt * L(u^(2)) ---
    initTimeStep(rk3.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)

    for p_idx in interior
        rk3.div3[p_idx] = rk3.gradientInterpolator(particleGrid, p_idx, rk3.rho_stage2, eq, settings)

        rho_final = (1/3) * rk3.rho_n[p_idx] + (2/3) * rk3.rho_stage2[p_idx] - (2/3) * dt * rk3.div3[p_idx]
        
        if !(rk3.fallbackInterpolator isa NoFallbackGrad) && rk3.mood(particleGrid, p_idx, rk3.rho_stage2, rho_final)
            rk3.div3[p_idx] = rk3.fallbackInterpolator(particleGrid, p_idx, rk3.rho_stage2, eq, settings; setCurvature=false)
            rho_final = (1/3) * rk3.rho_n[p_idx] + (2/3) * rk3.rho_stage2[p_idx] - (2/3) * dt * rk3.div3[p_idx]
        end
        particleGrid.rhos[p_idx] = rho_final
    end
end

struct RK4{G1, G2, MOOD} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_stage::Vector{Float64} # A single buffer for all intermediate stages
    
    k1::Vector{Float64} # Stores divergence from stage 1
    k2::Vector{Float64} # Stores divergence from stage 2
    k3::Vector{Float64} # Stores divergence from stage 3
    k4::Vector{Float64} # Stores divergence from stage 4

    function RK4(grad::G1, fallback::G2, mood::M) where {G1, G2, M}
        # Initialize with empty buffers; they will be resized on the first call
        new{G1, G2, M}(grad, fallback, mood, Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

# --- User-Friendly Constructor ---
function RK4(gradientInterpolator::G1; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1, G2, M}
    RK4(gradientInterpolator, fallbackInterpolator, mood)
end

function initTimeStepper(rk4::RK4, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !(rk4.fallbackInterpolator isa NoFallbackGrad)
        initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    end
end

function (rk4::RK4)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    #initMOOD!(rk4.mood,particleGrid.max_volume)
    # --- Ensure buffers are correctly sized for the current grid ---
    if length(rk4.rho_n) != N
        resize!.((rk4.rho_n, rk4.rho_stage, rk4.k1, rk4.k2, rk4.k3, rk4.k4), N)
    end

    interior = particleGrid.interior_indices
    rk4.rho_n .= particleGrid.rhos # Store u^n
    apply_boundary_conditions!(particleGrid)
    # --- Stage 1: Calculate k1 = -div(u^n) ---
    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    if !(rk4.fallbackInterpolator isa NoFallbackGrad); initTimeStep(rk4.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange); end

    for p_idx in interior
        rk4.k1[p_idx] = rk4.gradientInterpolator(particleGrid, p_idx, rk4.rho_n, eq, settings)
        
        # MOOD check is for the candidate solution of the *next* stage
        rho_candidate = rk4.rho_n[p_idx] - 0.5 * dt * rk4.k1[p_idx]
        if !(rk4.fallbackInterpolator isa NoFallbackGrad) && rk4.mood(particleGrid, p_idx, rk4.rho_n, rho_candidate; firstStage=true)
            rk4.k1[p_idx] = rk4.fallbackInterpolator(particleGrid, p_idx, rk4.rho_n, eq, settings; setCurvature=false)
        end
    end

    # --- Stage 2: Calculate k2 = -div(u^n + 0.5*dt*k1) ---
    @. rk4.rho_stage = rk4.rho_n - 0.5 * dt * rk4.k1
    particleGrid.rhos[interior] .= @view rk4.rho_stage[interior]
    apply_boundary_conditions!(particleGrid)
    rk4.rho_stage .= particleGrid.rhos # Update buffer with correct ghosts

    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for p_idx in interior
        rk4.k2[p_idx] = rk4.gradientInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings)
        rho_candidate = rk4.rho_n[p_idx] - 0.5 * dt * rk4.k2[p_idx]
        if !(rk4.fallbackInterpolator isa NoFallbackGrad) && rk4.mood(particleGrid, p_idx, rk4.rho_stage, rho_candidate)
            rk4.k2[p_idx] = rk4.fallbackInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings; setCurvature=false)
        end
    end

    # --- Stage 3: Calculate k3 = -div(u^n + 0.5*dt*k2) ---
    @. rk4.rho_stage = rk4.rho_n - 0.5 * dt * rk4.k2
    particleGrid.rhos[interior] .= @view rk4.rho_stage[interior]
    apply_boundary_conditions!(particleGrid)
    rk4.rho_stage .= particleGrid.rhos

    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for p_idx in interior
        rk4.k3[p_idx] = rk4.gradientInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings)
        rho_candidate = rk4.rho_n[p_idx] - dt * rk4.k3[p_idx]
        if !(rk4.fallbackInterpolator isa NoFallbackGrad) && rk4.mood(particleGrid, p_idx, rk4.rho_stage, rho_candidate)
            rk4.k3[p_idx] = rk4.fallbackInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings; setCurvature=false)
        end
    end

    # --- Stage 4: Calculate k4 = -div(u^n + dt*k3) ---
    @. rk4.rho_stage = rk4.rho_n - dt * rk4.k3
    particleGrid.rhos[interior] .= @view rk4.rho_stage[interior]
    apply_boundary_conditions!(particleGrid)
    rk4.rho_stage .= particleGrid.rhos

    initTimeStep(rk4.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for p_idx in interior
        rk4.k4[p_idx] = rk4.gradientInterpolator(particleGrid, p_idx, rk4.rho_stage, eq, settings)
        # Final MOOD check
        rho_final = rk4.rho_n[p_idx] - (dt/6) * (rk4.k1[p_idx] + 2*rk4.k2[p_idx] + 2*rk4.k3[p_idx] + rk4.k4[p_idx])
        if !(rk4.fallbackInterpolator isa NoFallbackGrad) && rk4.mood(particleGrid, p_idx, rk4.rho_stage, rho_final)
            # If the final step fails, a common fallback is to take a first-order Euler step
            # using the final stage's divergence (k4). This is a robust choice.
            particleGrid.rhos[p_idx] = rk4.rho_stage[p_idx] - dt * rk4.k4[p_idx]
        else
            particleGrid.rhos[p_idx] = rho_final
        end
    end
end


# No longer needs Nx, Ny. Buffers are sized based on the grid passed during the call.
struct RalstonRK2{G1, G2, MOOD} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    # Buffers are now part of the struct to be reused
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}

    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}

    function RalstonRK2(grad::G1, fallback::G2, mood::M) where {G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion}
        # Initialize with empty buffers, they will be resized on the first step
        new{G1, G2, M}(grad, fallback, mood, Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

# User-friendly constructor
function RalstonRK2(gradientInterpolator::G1; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1, G2, M}
    RalstonRK2(gradientInterpolator, fallbackInterpolator, mood)
end

function initTimeStepper(ralston::RalstonRK2, particleGrid::ParticleGrid, settings::SimSetting)
    updateNeighbors!(particleGrid, ralston.gradientInterpolator.weightFunction)
end


function (ralston::RalstonRK2)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    # --- Resize buffers only if necessary, using N ---
    if length(ralston.rhoInit) < N
        _ensure_capacity!(ralston.rhoInit, N)
        _ensure_capacity!(ralston.rhos, N)
        _ensure_capacity!(ralston.div1, N)
    end

    initGIBuffers!(ralston.gradientInterpolator, particleGrid)
    initGIBuffers!(ralston.fallbackInterpolator, particleGrid)


    # --- First RK Stage ---
    # 1. Start with the current, correct state of the grid
    ralston.rhoInit[1:N] .= particleGrid.rhos

    # 4. Apply boundary conditions to the intermediate result stored in the buffer
    apply_boundary_conditions!(particleGrid, ralston.rhoInit)

    initTS!(ralston, particleGrid, ralston.rhoInit)

    # Define a chunk size. 100 is a good starting point.
    chunk_size = 250 
    chunks = collect(Iterators.partition(1:N, chunk_size))

    # Partition 1:N into chunks of 100, and schedule *those* dynamically
    Threads.@threads :dynamic for particle_range in chunks
        for p_idx in particle_range
        fi = ralston.rhoInit[p_idx]
        num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w = getNBInput(particleGrid, p_idx, ralston.neighbor_fs, ralston.neighbor_dfs)
            initGI!(ralston.gradientInterpolator, p_idx, fi, num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w)
            initGI!(ralston.fallbackInterpolator, p_idx, fi, num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w)
        end
    end
    
    # 3. Calculate divergence for interior particles
    Threads.@threads :dynamic for particle_range in chunks
        for p_idx in particle_range
            if particleGrid.is_boundary[p_idx]; continue; end # Skip ghost particles
            fi = ralston.rhoInit[p_idx]
            num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w = getNBInput(particleGrid, p_idx, ralston.neighbor_fs, ralston.neighbor_dfs)

            ralston.div1[p_idx] = ralston.gradientInterpolator(eq, p_idx, fi, num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w)
            
            rho_candidate = ralston.rhoInit[p_idx] - ralston.div1[p_idx] * dt * 2/3
            
            if !(ralston.fallbackInterpolator isa NoFallbackGrad) && ralston.mood(particleGrid, p_idx, ralston.rhoInit, rho_candidate; firstStage=true)
                ralston.div1[p_idx] = ralston.fallbackInterpolator(p_idx, fi, num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w)
                rho_candidate = ralston.rhoInit[p_idx] - ralston.div1[p_idx] * dt * 2/3
            end
            ralston.rhos[p_idx] = rho_candidate # Store intermediate result in the 'rhos' buffer
        end
    end
    # 4. Apply boundary conditions to the intermediate result stored in the buffer
    apply_boundary_conditions!(particleGrid, ralston.rhos)

    initTS!(ralston, particleGrid, ralston.rhos)

    # Partition 1:N into chunks of 100, and schedule *those* dynamically
    Threads.@threads :dynamic for particle_range in chunks
        for p_idx in particle_range
        fi = ralston.rhos[p_idx]
        num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w = getNBInput(particleGrid, p_idx, ralston.neighbor_fs, ralston.neighbor_dfs)
            initGI!(ralston.gradientInterpolator, p_idx, fi, num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w)
            initGI!(ralston.fallbackInterpolator, p_idx, fi, num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w)
        end
    end

    # 2. Calculate final divergence for interior particles
    Threads.@threads :dynamic for particle_range in chunks
        for p_idx in particle_range
            if particleGrid.is_boundary[p_idx]; continue; end # Skip ghost particles
        
            fi = ralston.rhoInit[p_idx]
            num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w = getNBInput(particleGrid, p_idx, ralston.neighbor_fs, ralston.neighbor_dfs)

            # Pass the intermediate state (ralston.rhos) to the gradient calculation
            div2 = ralston.gradientInterpolator(eq, p_idx, fi, num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w)
            
            rho_final = ralston.rhoInit[p_idx] - dt * (ralston.div1[p_idx] / 4 + 3 * div2 / 4)
            if !(ralston.fallbackInterpolator isa NoFallbackGrad) && ralston.mood(particleGrid, p_idx, ralston.rhos, ralston.neighbor_fs, rho_final)
                div2 = ralston.fallbackInterpolator(p_idx, fi, num_nb, neighbor_slice, neighbors, f_neighbors, df_neighbors, dx, dy, w)
                rho_final = ralston.rhoInit[p_idx] - dt * (ralston.div1[p_idx] / 4 + 3 * div2 / 4)
            end
            # Directly write the final result for this particle into the grid
            particleGrid.rhos[p_idx] = rho_final
        end
    end
    # Final boundary condition application after the full step is complete
    apply_boundary_conditions!(particleGrid, particleGrid.rhos)
    
end


struct RalstonRK2SmoothSwitch{G1, G2, MOOD} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    tol::Float64
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_stage::Vector{Float64}
    rho_fallback::Vector{Float64}
    div1::Vector{Float64}
    
    # --- Propagation Buffers ---
    mood_indices::Vector{Int}
    prop_indices::Vector{Int}
    
    # Per-step flag to track which particles have been switched to fallback
    switched_to_fallback::BitVector

    function RalstonRK2SmoothSwitch(grad::G1, fallback::G2, mood::M; tol=1e-7) where {G1, G2, M}
        # Initialize with empty buffers; they will be resized on the first call
        new{G1, G2, M}(grad, fallback, mood, tol,
            Float64[], Float64[], Float64[], Float64[], # Main buffers
            Int[], Int[], # Propagation buffers
            falses(0)    # Flag buffer
        )
    end
end

# --- User-Friendly Constructor ---
function RalstonRK2SmoothSwitch(gradientInterpolator::G1; fallbackInterpolator::G2 = gradientInterpolator, mood::M = NoMOOD(), tol = 1e-7) where {G1, G2, M}
    RalstonRK2SmoothSwitch(gradientInterpolator, fallbackInterpolator, mood; tol=tol)
end

function initTimeStepper(ralston::RalstonRK2SmoothSwitch, particleGrid::ParticleGrid, settings::SimSetting)
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
end

function (ralston::RalstonRK2SmoothSwitch)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = particleGrid.N
    #initMOOD!(ralston.mood,particleGrid.max_volume)
    # --- Ensure buffers are correctly sized for the current grid ---
    if length(ralston.rho_n) != N
        resize!.((ralston.rho_n, ralston.rho_stage, ralston.rho_fallback, ralston.div1), N)
        resize!(ralston.switched_to_fallback, N)
    end

    interior = particleGrid.interior_indices
    ralston.rho_n .= particleGrid.rhos # Store u^n
    apply_boundary_conditions!(particleGrid)
    # --- 1. Calculate Full Fallback Solution and Target Mass ---
    initTimeStep(ralston.fallbackInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    target_mass = 0.0
    for p_idx in interior
        div_fallback = ralston.fallbackInterpolator(particleGrid, p_idx, ralston.rho_n, eq, settings; setCurvature=false)
        ralston.rho_fallback[p_idx] = ralston.rho_n[p_idx] - div_fallback * dt
        target_mass += ralston.rho_fallback[p_idx] * particleGrid.volumes[p_idx]
    end

    # --- 2. Perform High-Order RalstonRK2 Step ---
    # First stage
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    for p_idx in interior
        ralston.div1[p_idx] = ralston.gradientInterpolator(particleGrid, p_idx, ralston.rho_n, eq, settings)
        ralston.rho_stage[p_idx] = ralston.rho_n[p_idx] - ralston.div1[p_idx] * dt * 2/3
    end
    particleGrid.rhos[interior] .= @view ralston.rho_stage[interior]
    apply_boundary_conditions!(particleGrid)
    ralston.rho_stage .= particleGrid.rhos

    # Final stage
    initTimeStep(ralston.gradientInterpolator, particleGrid, settings.interpAlpha, settings.interpRange)
    current_mass = 0.0
    empty!(ralston.mood_indices)
    fill!(ralston.switched_to_fallback, false)

    for p_idx in interior
        div2 = ralston.gradientInterpolator(particleGrid, p_idx, ralston.rho_stage, eq, settings)
        rho_final_candidate = ralston.rho_n[p_idx] - dt * (ralston.div1[p_idx]/4 + 3*div2/4)
        
        # --- 3. Initial MOOD Check ---
        if ralston.mood(particleGrid, p_idx, ralston.rho_n, rho_final_candidate; firstStage=true)
            particleGrid.rhos[p_idx] = ralston.rho_fallback[p_idx]
            push!(ralston.mood_indices, p_idx)
            ralston.switched_to_fallback[p_idx] = true
        else
            particleGrid.rhos[p_idx] = rho_final_candidate
        end
        current_mass += particleGrid.rhos[p_idx] * particleGrid.volumes[p_idx]
    end

    # --- 4. Mass Conservation Propagation Loop ---
    if !isempty(ralston.mood_indices)
        # Build the initial propagation list from neighbors of MOOD events
        empty!(ralston.prop_indices)
        for p_idx in ralston.mood_indices
            for nb_idx in particleGrid.neighbour_indices[p_idx]
                # Only add interior neighbors that haven't been switched yet
                if nb_idx in interior && !ralston.switched_to_fallback[nb_idx]
                    push!(ralston.prop_indices, nb_idx)
                end
            end
        end
        unique!(ralston.prop_indices) # Remove duplicates

        while abs(target_mass - current_mass) > ralston.tol && !isempty(ralston.prop_indices)
            p_idx = popfirst!(ralston.prop_indices)
            
            # This check is redundant if we filter when adding, but safe
            if ralston.switched_to_fallback[p_idx]; continue; end
            
            # Switch this particle to the low-order solution
            local_mass_change = (ralston.rho_fallback[p_idx] - particleGrid.rhos[p_idx]) * particleGrid.volumes[p_idx]
            current_mass += local_mass_change
            particleGrid.rhos[p_idx] = ralston.rho_fallback[p_idx]
            ralston.switched_to_fallback[p_idx] = true

            # Add its neighbors to the propagation list
            for nb_idx in particleGrid.neighbour_indices[p_idx]
                if nb_idx in interior && !ralston.switched_to_fallback[nb_idx]
                    push!(ralston.prop_indices, nb_idx)
                end
            end
        end
    end
end




