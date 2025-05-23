
# Default case: Everything is decoupled
export RelaxationStepper

function (ts::TimeStepper)(eqs::Vector{<:ScalarHyperbolicEquation}, particleGrids::Vector{<:ParticleGrid}, settings::SimSetting, time::Real, dt::Real)
    @warn "No dedicated system timestepper found. System will be treated independently using the given scalar timestepper."
    for (i,particleGrid) = enumerate(particleGrids)
        ts(eqs[i], particleGrid, settings, time, dt)
    end
end

function (ts::TimeStepper)(eq::ScalarHyperbolicEquation, particleGrids::Vector{<:ParticleGrid}, settings::SimSetting, time::Real, dt::Real)
    @warn "Only one scalar hyperbolic equation for a system is found. The scalar equation will be used for all components!"
    ts([eq for _ = eachindex(particleGrids)], particleGrids, settings, time, dt)
end

struct RelaxationStepper <: MeshfreeSystemTimeStepper
    timestepper::TimeStepper
    M::Vector{T} where T <: Function
    epsilon::Float64
    rho::Vector{Float64}

    function RelaxationStepper(timestepper::TimeStepper, Nx::Int64, M::Vector{T}; epsilon = 10. ^-10) where T <: Function
        @assert !isa(timestepper, MeshfreeSystemTimeStepper) "A scalar timestepper has to be given!"
        new(timestepper, M, epsilon, Vector{Float64}(undef, Nx))
    end
end

function initTimeStepper(method::RelaxationStepper, particleGrids::Vector{T}, settings::SimSetting) where T <: ParticleGrid
    initTimeStepper(method.timestepper, particleGrids[1], settings)
end

function(relax_ts::RelaxationStepper)(eqs::Vector{LinearAdvection{T}}, particleGrids::Vector{<:ParticleGrid}, settings::SimSetting, time::Real, dt::Real) where T <: Float64
    for (i,pg) = enumerate(particleGrids)
        relax_ts.timestepper(eqs[i], pg, settings, time, dt)
    end
    relax_ts.rho[:] = sum([[p.rho for p = pg.grid] for pg = particleGrids])
    for (i_pg,pg) = enumerate(particleGrids)
        
        for (i_p,p) = enumerate(pg.grid)
            p.rho = relax_ts.epsilon/(relax_ts.epsilon + dt) * p.rho + dt/(relax_ts.epsilon+dt) * relax_ts.M[i_pg](relax_ts.rho[i_p])
        end
    end

end

# In your MeshfreeTimeSteppers.txt (or wherever RelaxationStepper is defined)

# struct RelaxationStepper <: MeshfreeSystemTimeStepper ... end
# function RelaxationStepper(...) ... end

# function(relax_ts::RelaxationStepper)(
#     eqs::Vector{<:LinearAdvection}, # Vector of LinearAdvection equations for v_k
#     particleGrids::Vector{<:ParticleGrid}, # This is your `pgs`, one for each v_k
#     settings::SimSetting,
#     time::Real,
#     dt::Real
# )
#     N_particles = length(particleGrids[1].grid) # Assume all component grids have same N
#     N_components = length(particleGrids)

#     # Ensure relax_ts.rho is correctly sized (was Vector{Float64}(undef, Nx) in struct)
#     # Nx in struct constructor should match N_particles
#     if length(relax_ts.rho) != N_particles
#         # This indicates a mismatch in construction or grid size passed.
#         # For safety, one might resize, but ideally, it's constructed correctly.
#         # error("relax_ts.rho size mismatch with particle grid size")
#         resize!(relax_ts.rho, N_particles)
#     end

#     # --- Step 1: Advection Step (Explicit) ---
#     # This step updates each particleGrids[k_comp].grid[p_idx].rho to v_k^*(p_idx)
#     # Store these v_k^* values before they are overwritten by the relaxation step if needed,
#     # or ensure the relaxation step uses the correct post-advection values.

#     # Let's create a temporary store for post-advection values v_k^*
#     # to avoid using partially updated values in the sum for rho_total_star
#     v_star_components = Matrix{Float64}(undef, N_particles, N_components)

#     for k_comp in 1:N_components
#         pg_k = particleGrids[k_comp] # Current component's grid
#         eq_k = eqs[k_comp]           # Current component's advection equation

#         # The scalar timestepper relax_ts.timestepper updates pg_k.grid[p].rho in place.
#         # It uses the rho values from the *previous full time step* (stored in its internal rhoOld)
#         # and updates pg_k.grid[p].rho to the new value after this scalar advection substep.
#         relax_ts.timestepper(eq_k, pg_k, settings, time, dt) # Modifies pg_k.grid[p].rho

#         # After this call, pg_k.grid[p_idx].rho contains v_k^*(p_idx)
#         for p_idx in 1:N_particles
#             v_star_components[p_idx, k_comp] = pg_k.grid[p_idx].rho
#         end
#     end

#     # --- Step 2: Calculate total "density" rho_total_star = sum_k v_k^* ---
#     # This sum is performed at each particle location.
#     # relax_ts.rho will store rho_total_star for each particle.
#     fill!(relax_ts.rho, 0.0) # Initialize sum to zero
#     for p_idx in 1:N_particles
#         for k_comp in 1:N_components
#             relax_ts.rho[p_idx] += v_star_components[p_idx, k_comp]
#         end
#     end
#     # Now, relax_ts.rho[p_idx] = sum_over_k( v_k^*(p_idx) )

#     # --- Step 3: Relaxation Step ---
#     # Updates particleGrids[k_comp].grid[p_idx].rho to v_k^{n+1}
#     # v_k^{n+1} = [epsilon/(epsilon+dt)] * v_k^* + [dt/(epsilon+dt)] * M_k(rho_total_star)
#     for k_comp in 1:N_components
#         pg_k = particleGrids[k_comp] # Current component's grid
#         maxwellian_func_k = relax_ts.M[k_comp] # M_k

#         coeff_ep = relax_ts.epsilon / (relax_ts.epsilon + dt)
#         coeff_dt = dt / (relax_ts.epsilon + dt)

#         for p_idx in 1:N_particles
#             v_k_star_at_p = v_star_components[p_idx, k_comp] # Value of this component after advection
#             rho_total_star_at_p = relax_ts.rho[p_idx]      # Sum of all components after advection

#             equilibrium_val_k = maxwellian_func_k(rho_total_star_at_p)
            
#             pg_k.grid[p_idx].rho = coeff_ep * v_k_star_at_p + coeff_dt * equilibrium_val_k
#         end
#     end
# end