
# Default case: Everything is decoupled
export GeneralIMEXTimeStepper, ARS233, PareschiRussoIMEXSSP3, ARS222, SSP2332, SimpleSplitting, RalstonRK2, ARS232

include("ButcherTableaus.jl")

# --- In your TimeIntegration.jl file ---

using InteractiveUtils

function run_functor_test(rs, S_out, U_kin, pos, time)
    # This is the call we want to inspect
    rs(S_out, U_kin, pos, time)
end

# In your main code, after `my_source_term` is defined:
# 

#==============================================================================
  Simple Splitting Timestepper (Optimized for SoA Grids)
==============================================================================#

# --- Generic System Stepper Fallbacks (as provided by you) ---
function (ts::TimeStepper)(eqs::DiagonalHyperbolicSystem{N,D}, particleGrids::ParticleGridSystem{N}, settings::SimSetting, time::Real, dt::Real) where {N,D}
    @warn "No dedicated system timestepper found. System will be treated independently using the given scalar timestepper."
    for (i,particleGrid) = enumerate(particleGrids)
        ts(eqs[i], particleGrid, settings, time, dt)
    end
end

function (ts::TimeStepper)(eq::ScalarHyperbolicPDE, particleGrids::ParticleGridSystem{N}, settings::SimSetting, time::Real, dt::Real) where {N}
    @warn "Only one scalar hyperbolic equation for a system is found. The scalar equation will be used for all components!"
    ts(ntuple(i -> eq, N), particleGrids, settings, time, dt)
end


# --- REFACTORED SimpleSplitting Struct and Constructor ---
mutable struct SimpleSplitting{T <: TimeStepper, S <: AbstractSourceTerm} <: MeshfreeSystemTimeStepper
    timestepper::T 
    source_term::S
    macro_state_buffer::Matrix{Float64} 

    function SimpleSplitting(
        timestepper::T, 
        source_term::S
    ) where {T <: TimeStepper, S <: AbstractSourceTerm}
        @assert !isa(timestepper, MeshfreeSystemTimeStepper) "A scalar timestepper must be provided for the advection step."
        @assert isa(source_term, RelaxationSourceTerm) "Source term must be a RelaxationSourceTerm."
        
        # Initialize buffer as an empty matrix. It will be resized on the first time step.
        macro_buffer = Matrix{Float64}(undef, 0, 0)
        
        new{T, S}(timestepper, source_term, macro_buffer)
    end
end

function initTimeStepper(method::SimpleSplitting, particleGrids::ParticleGridSystem, settings::SimSetting)
    # Initialize the underlying scalar timestepper for each component grid
    for pg in particleGrids
        initTimeStepper(method.timestepper, pg, settings)
    end
end

"""
Functor for the SimpleSplitting timestepper, optimized for SoA grids.
"""
function (ss::SimpleSplitting)(
    eqs::DiagonalHyperbolicSystem{N,D},
    particleGrids::ParticleGridSystem{N}, 
    settings::SimSetting, 
    time::Real, 
    dt::Real
) where {N,D}
    
    # --- 1. Advection Step ---
    # Apply the scalar timestepper to each kinetic component.
    # This updates the `rhos` array in each grid to the post-advection state v_k^*.
    for k_comp in 1:N
        ss.timestepper(eqs[k_comp], particleGrids[k_comp], settings, time, dt)
    end

    # --- 2. Recombination Step ---
    N_total_particles = particleGrids[1].N
    rs = ss.source_term

    # --- NEW: Ensure the buffer is correctly sized ---
    if size(ss.macro_state_buffer, 1) != N_total_particles || size(ss.macro_state_buffer, 2) != rs.num_macro_variables
        ss.macro_state_buffer = Matrix{Float64}(undef, N_total_particles, rs.num_macro_variables)
    end

    # Reconstruct the macroscopic state U_macro^* at ALL particle locations
    for p_idx in 1:N_total_particles
        for i_macro in 1:rs.num_macro_variables
            # Sum the advected kinetic variables (v_k^*) using direct array access
            indices = rs.kinetic_indices[i_macro]
            ss.macro_state_buffer[p_idx, i_macro] = sum(particleGrids[k_idx].rhos[p_idx] for k_idx in indices)
        end
    end

    # --- 3. Relaxation Step ---
    coeff_ep = rs.epsilon / (rs.epsilon + dt)
    coeff_dt = dt / (rs.epsilon + dt)

    for k_comp in 1:N
        pg_k = particleGrids[k_comp]
        maxwellian_func_k = rs.maxwellians[k_comp]

        for p_idx in pg_k.interior_indices
            # v_k^*(p_idx) from the advection step
            v_k_star_at_p = pg_k.rhos[p_idx]
            
            # Get the macroscopic state vector (U_macro_i^*) using ntuple for performance
            U_macro_star_at_p = ntuple(i -> ss.macro_state_buffer[p_idx, i], rs.num_macro_variables)

            # Evaluate the Maxwellian (it expects a tuple)
            equilibrium_val_k = maxwellian_func_k(U_macro_star_at_p)
            
            # Update the rhos array to the final state v_k^{n+1}
            pg_k.rhos[p_idx] = coeff_ep * v_k_star_at_p + coeff_dt * equilibrium_val_k
        end
    end
end
# --- In your TimeIntegration.jl file ---
# --- REFACTORED GeneralIMEXTimeStepper Struct and Constructor ---
mutable struct GeneralIMEXTimeStepper{G1, G2, M, IS, ST_OBJ, BT} <: MeshfreeSystemTimeStepper
    # User's modular components
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::M
    implicit_solver::IS
    source_term_object::ST_OBJ
    butcher_tableau::BT
    
    # --- Reusable Buffers (Workspace) ---
    U_n_sys::Matrix{Float64}
    Y_stages_sys::Vector{Matrix{Float64}}
    K_E_stages_sys::Vector{Matrix{Float64}}
    K_I_stages_sys::Vector{Matrix{Float64}}
    rho_buffer::Vector{Float64} # Scalar buffer (size N_particles)
    
    # --- Particle-local buffers for implicit solve ---
    u_particle_iter_buffer::Vector{Float64} # (size N_components)
    Y_i_base_particle_buffer::Vector{Float64} # (size N_components)
    
    # --- Buffers for explicit fused loop (like in RK4) ---
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}
    
    num_stages::Int

    function GeneralIMEXTimeStepper(
            gradientInterpolator::G1, fallbackInterpolator::G2, mood::M,
            implicit_solver::IS, source_term_object::ST_OBJ, butcher_tableau::BT
        ) where {G1, G2, M, IS, ST_OBJ, BT}
        
        s = size(butcher_tableau.A, 1) # Number of stages

        # Initialize with empty buffers; they will be resized on the first call
        new{G1, G2, M, IS, ST_OBJ, BT}(
            gradientInterpolator, fallbackInterpolator, mood, implicit_solver, 
            source_term_object, butcher_tableau,
            Matrix{Float64}(undef,0,0), [Matrix{Float64}(undef,0,0) for _ in 1:s],
            [Matrix{Float64}(undef,0,0) for _ in 1:s], [Matrix{Float64}(undef,0,0) for _ in 1:s], 
            Vector{Float64}(undef, 0), # rho_buffer
            Float64[], Float64[], # implicit buffers
            Float64[], Float64[], # neighbor_fs, neighbor_dfs
            s
        )
    end
end

# --- initTimeStepper is REMOVED (part of old structure) ---

# --- NEW: initAddTSBuffer! for the IMEX stepper ---
# This resizes the buffers that are per-particle, but not system-wide
function initAddTSBuffer!(imex_ts::GeneralIMEXTimeStepper, pg::ParticleGrid)
    num_particles = length(pg.num_neighbors)
    _ensure_capacity!(imex_ts.rho_buffer, num_particles)
    
    # Note: System-wide matrices (N_particles x N_components) and
    # component-wide vectors (N_components) are resized inside the main functor,
    # as this is the only place that knows both N_particles and N_components.
end


# --- REFACTORED Functor for GeneralIMEXTimeStepper ---
function (imex_ts::GeneralIMEXTimeStepper{G1, G2, M, IS, ST_OBJ, BT})(
        scalar_equations::DiagonalHyperbolicSystem{N,D},
        system_pg::ParticleGridSystem{N},
        settings::SimSetting,
        time_n::Real,
        dt::Real
    ) where {G1, G2, M, IS, ST_OBJ, BT, N, D}

    N_particles = system_pg[1].N
    N_components = N
    s = imex_ts.num_stages
    bt = imex_ts.butcher_tableau
    
    # Define chunks for parallel loops
    chunk_size = 50 
    chunks = collect(Iterators.partition(1:N_particles, chunk_size))

    # --- Ensure buffers are correctly sized for the current grid ---
    if size(imex_ts.U_n_sys, 1) != N_particles
        # --- System-wide (N_particles x N_components) ---
        imex_ts.U_n_sys = Matrix{Float64}(undef, N_particles, N_components)
        for i in 1:s
            imex_ts.Y_stages_sys[i] = Matrix{Float64}(undef, N_particles, N_components)
            imex_ts.K_E_stages_sys[i] = Matrix{Float64}(undef, N_particles, N_components)
            imex_ts.K_I_stages_sys[i] = Matrix{Float64}(undef, N_particles, N_components)
        end
        
        # --- Per-component (size N_components) ---
        resize!(imex_ts.u_particle_iter_buffer, N_components)
        resize!(imex_ts.Y_i_base_particle_buffer, N_components)
        
        # --- Per-interaction (size M) ---
        # (This will be resized by initTSBuffer! inside the loop)
    end

    # --- 0. Store U^n from system_pg ---
    for k in 1:N; imex_ts.U_n_sys[:, k] = system_pg[k].rhos; end

    # --- Loop through stages i = 1 to s ---
    for i in 1:s
        current_Y_i_sys = imex_ts.Y_stages_sys[i]
        current_Y_i_sys .= imex_ts.U_n_sys # Start with U^n

        # --- Calculate stage value Y_i for INTERIOR points ---
        # (This part is sequential and remains unchanged)
        for j in 1:(i-1)
            if bt.At[i,j] != 0.0; @. current_Y_i_sys[1:N_particles, :] += dt * bt.At[i,j] * imex_ts.K_E_stages_sys[j][1:N_particles, :]; end
            if bt.A[i,j] != 0.0;  @. current_Y_i_sys[1:N_particles, :] += dt * bt.A[i,j] * imex_ts.K_I_stages_sys[j][1:N_particles, :]; end
        end
        
        # --- Implicit Solve for stage Y_i for INTERIOR points ---
        # (This part is sequential and remains unchanged)
        if abs(bt.A[i,i]) > 1e-14
            time_implicit = time_n + bt.c[i] * dt
            for p_idx in 1:N_particles # (Assuming implicit solve on all)
                imex_ts.u_particle_iter_buffer .= @view current_Y_i_sys[p_idx, :] 
                imex_ts.Y_i_base_particle_buffer .= @view current_Y_i_sys[p_idx, :] 
                                          
                ImplicitSolvers.solve!(imex_ts.implicit_solver,
                    imex_ts.u_particle_iter_buffer, imex_ts.Y_i_base_particle_buffer, dt * bt.A[i,i],
                    imex_ts.source_term_object, system_pg[1].positions[p_idx], time_implicit, N_components
                )
                current_Y_i_sys[p_idx, :] .= imex_ts.u_particle_iter_buffer
            end
        end
        
        # --- Ghost Cell Update for Intermediate Stage Y_i ---
        # (This part is sequential and remains unchanged)
        for k in 1:N
            apply_boundary_conditions!(system_pg[k],@view current_Y_i_sys[:, k])
        end
        
        # ==================================================================
        # --- REFACTORED: Evaluate and store explicit tendency K_E ---
        # ==================================================================
        
        # Loop over each component (equation) in the system
        for k in 1:N
            eq_k = scalar_equations[k]
            grid_k = system_pg[k]
            # Get the view of the current stage for this component
            current_Y_i_k = @view current_Y_i_sys[:, k]

            # 1. Init Buffers for this component
            initGIBuffers!(imex_ts.gradientInterpolator, grid_k)
            initGIBuffers!(imex_ts.fallbackInterpolator, grid_k)
            initTSBuffer!(imex_ts, grid_k) # Resizes neighbor_fs/dfs

            # 2. Threaded loop to calculate slopes/coefficients
            Threads.@threads for particle_range in chunks
                for p_idx in particle_range
                    fi = current_Y_i_k[p_idx]
                    initFs!(imex_ts, p_idx, fi, current_Y_i_k, grid_k)
                    initGI!(imex_ts.gradientInterpolator, p_idx, fi, grid_k, imex_ts.neighbor_fs, imex_ts.neighbor_dfs)
                    initGI!(imex_ts.fallbackInterpolator, p_idx, fi, grid_k, imex_ts.neighbor_fs, imex_ts.neighbor_dfs)
                end
            end

            # 3. Threaded loop to calculate divergence
            Threads.@threads for particle_range in chunks
                for p_idx in particle_range
                    if grid_k.is_boundary[p_idx]; continue; end

                    fi = current_Y_i_k[p_idx]
                    nb_slice = getNBSlice(grid_k, p_idx)
                    
                    div_high = imex_ts.gradientInterpolator(eq_k, p_idx, fi, nb_slice, grid_k, imex_ts.neighbor_fs, imex_ts.neighbor_dfs)
                    
                    rho_candidate = fi - dt * div_high # Candidate for MOOD
                    
                    if !(imex_ts.fallbackInterpolator isa NoFallbackGrad) && imex_ts.mood(imex_ts.gradientInterpolator, p_idx, fi, nb_slice, rho_candidate, grid_k, current_Y_i_k)
                        div_fallback = imex_ts.fallbackInterpolator(eq_k, p_idx, fi, nb_slice, grid_k, imex_ts.neighbor_fs, imex_ts.neighbor_dfs)
                        imex_ts.K_E_stages_sys[i][p_idx, k] = -div_fallback
                    else
                        imex_ts.K_E_stages_sys[i][p_idx, k] = -div_high
                    end
                end
            end
        end # End of component loop for K_E
        
        # ==================================================================
        # --- Evaluate and store implicit tendency K_I ---
        # ==================================================================
        # (This part is sequential and remains unchanged)
        time_implicit_for_KI = time_n + bt.c[i] * dt 
        
        for p_idx in 1:N_particles
            imex_ts.source_term_object(
                @view(imex_ts.K_I_stages_sys[i][p_idx, :]), 
                @view(current_Y_i_sys[p_idx, :]), 
                system_pg[1].positions[p_idx], 
                time_implicit_for_KI
            )
        end
    end # End of stages loop

    # --- Final Update ---
    # (This part is sequential and remains unchanged)
    U_np1 = imex_ts.U_n_sys # Reuse this buffer for the final result
    for i in 1:s
        if abs(bt.bt[i]) > 1e-14; U_np1[1:N_particles, :] .+= dt * bt.bt[i] .* @view(imex_ts.K_E_stages_sys[i][1:N_particles, :]); end
        if abs(bt.b[i]) > 1e-14;  U_np1[1:N_particles, :] .+= dt * bt.b[i] .* @view(imex_ts.K_I_stages_sys[i][1:N_particles, :]); end
    end

    # --- Update physical particle grids ---
    for k in 1:N
        system_pg[k].rhos .= @view U_np1[:, k]
        # system_pg[k].mood_events .= false # (If you have this field)
    end
end

"""
    ARS233(
        gradientInterpolator, 
        fallbackInterpolator, # Union{GradientInterpolator, Nothing}
        mood_criterion,
        implicit_solver, 
        source_term_object,
        N_particles::Int, 
        N_components::Int;
        gamma_coefficient::Float64 = (3.0 + sqrt(3.0))/6.0 # Specific to ARS233
    )

Constructs a `GeneralIMEXTimeStepper` pre-configured with the IMEXARS233 Butcher tableau.
This scheme is a 3-stage, 3rd order Additive Runge-Kutta scheme.
"""
function ARS233( # Changed name to avoid conflict with potential struct name if desired
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
    gamma_coefficient::Float64 = (3.0 + sqrt(3.0))/6.0 # Allow custom gamma for this specific scheme
) where {
    G1 <: Interpolations.GradientInterpolator, # Example: Qualify with your module name
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    # Get the specific Butcher tableau for IMEXARS233
    # Pass the gamma if your tableau function accepts it.
    tableau = IMEXARS233ButcherTableau(gamma_coefficient) 

    return GeneralIMEXTimeStepper( # Assuming GeneralIMEXTimeStepper is defined in current scope
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific Butcher tableau
    )
end
# In a file like IMEXTableaus.jl or alongside GeneralIMEXTimeStepper definition
# Ensure your IMEXButcherTableau struct is defined as you provided previously.
# struct IMEXButcherTableau{M <: AbstractArray{Float64, 2}, V <: AbstractArray{Float64, 1}} ... end

"""
    PR_IMEX_SSP3_ButcherTableau()::IMEXButcherTableau

Returns the Butcher tableau for the Pareschi & Russo (2005) IMEX-SSP3(3,3,3)
scheme (Scheme 4.2 in their JCP paper "Implicit-explicit Runge-Kutta schemes
and applications to hyperbolic systems with relaxation").
This is a 3-stage, 3rd order, L-stable scheme.
"""


# In your MeshfreeTimeSteppers.jl or SystemIMEXTimeSteppers.jl
# Ensure GeneralIMEXTimeStepper and all necessary types (GradientInterpolator, etc.)
# and PR_IMEX_SSP3_ButcherTableau are accessible.

"""
    PareschiRussoIMEXSSP3(
        gradientInterpolator, fallbackInterpolator, mood_criterion,
        implicit_solver, source_term_object,
        N_particles::Int, N_components::Int
    )

Constructs a GeneralIMEXTimeStepper pre-configured with the Pareschi & Russo (2005)
IMEX-SSP3(3,3,3) Butcher tableau (Scheme 4.2).
This is a 3-stage, 3rd order, L-stable scheme.
"""
function PareschiRussoIMEXSSP3(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
) where {
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    tableau = PR_IMEX_SSP3_ButcherTableau() 

    return GeneralIMEXTimeStepper(
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific Butcher tableau
    )
end

"""
    ARS222_IMEX(
        gradientInterpolator, fallbackInterpolator, mood_criterion,
        implicit_solver, source_term_object,
        N_particles::Int, N_components::Int;
        gamma_coefficient::Union{Float64,Nothing}=nothing 
    )

Constructs a GeneralIMEXTimeStepper pre-configured with the standard
ARS(2,2,2) IMEX Butcher tableau by Ascher, Ruuth, Spiteri (1997).
This is a 2-stage, 2nd order, L-stable scheme.
"""
function ARS222(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
    gamma_coefficient::Union{Float64,Nothing}=nothing # Allows override of default gamma
) where {
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    tableau = ARS222_ButcherTableau(gamma_coefficient) 

    return GeneralIMEXTimeStepper(
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific ARS(2,2,2) Butcher tableau
    )
end


function SSP2332(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
) where {
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    tableau = SSP2332ButcherTableau() 

    return GeneralIMEXTimeStepper(
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific ARS(2,2,2) Butcher tableau
    )
end

function RalstonRK2(
    gradientInterpolator::G1,
    fallbackInterpolator::G2,
    mood_criterion::M,
    implicit_solver::IS,
    source_term_object::ST_OBJ,
) where {
    G1 <: Interpolations.GradientInterpolator,
    G2 <: Union{Interpolations.GradientInterpolator, Nothing},
    M <: MOODCriterion, # Assuming MOODCriterion is defined
    IS <: ImplicitSolvers.AbstractImplicitSolver,
    ST_OBJ <: SourceTerms.AbstractSourceTerm
}
    
    tableau = RalstonRK2ButcherTableau() 

    return GeneralIMEXTimeStepper(
        gradientInterpolator,
        fallbackInterpolator,
        mood_criterion,
        implicit_solver,
        source_term_object,
        tableau, # The specific ARS(2,2,2) Butcher tableau
    )
end