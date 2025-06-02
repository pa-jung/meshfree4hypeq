module PlottingUtils

using ..ParticleGrids
using ..Particles
using ..SimSettings
using ..Interpolations
using IPlotPDESols

export calculateStats, calculateAllStats!

function calculateStats(
    u_numerical::AbstractVector{<:Real},
    u_analytical_func::Function,
    x_coords::AbstractVector{<:Real},
    xmin_domain::Real,
    xmax_domain::Real,
    N_particles::Int; # Make sure this is consistent with lengths of x_coords and u_numerical
    settings::SimSettings.SimSetting,
    order::Int = 0, 
    stats_to_calculate::Vector{String} = ["l1norm", "l2norm", "supnorm", "mass", "volume"]
)::Dict{String, Float64}

    if N_particles == 0
        @warn "Numerical solution vector is empty based on N_particles. Returning empty stats."
        return Dict{String, Float64}()
    end
    if length(u_numerical) != N_particles || length(x_coords) != N_particles
        error("Input u_numerical and x_coords lengths ($(length(u_numerical)), $(length(x_coords))) must match N_particles ($N_particles).")
    end

    results = Dict{String, Float64}()

    # --- 1. Create and Initialize Temporary ParticleGrid1D ---
    temp_particles = [Particles.Particle1D(x_coords[i], u_numerical[i], false) for i in 1:N_particles]
    if !isempty(temp_particles)
        temp_particles[1].boundary = true # As per your ParticleGrid1D constructor constraint
    end
    
    nominal_dx = (xmax_domain - xmin_domain) / (N_particles > 0 ? N_particles : 1)
    is_regular_grid = true # Determine this robustly if necessary for your ParticleGrid constructor
    if N_particles > 1
        for i_reg in 2:N_particles
            if !isapprox(x_coords[i_reg] - x_coords[i_reg-1], nominal_dx, atol=1e-9 * abs(nominal_dx))
                is_regular_grid = false; break
            end
        end
    end
    
    # Use the ParticleGrid1D constructor that takes a vector of particles and other geometric info
    temp_grid = ParticleGrids.ParticleGrid1D(temp_particles, xmin_domain, xmax_domain, nominal_dx, is_regular_grid)
    
    determineVolumes!(temp_grid) # YOU PROVIDE THIS FUNCTION - it populates particle.volume
    ParticleGrids.updateNeighbours!(temp_grid, settings.interpRange) # Populates particle.neighbourIndices

    # --- 2. Calculate Analytical Solution and Errors at particle locations ---
    u_analytical_at_particles = Vector{Float64}(undef, N_particles)
    for i in 1:N_particles
         # Assuming u_analytical_func takes only x for 1D
        u_analytical_at_particles[i] = u_analytical_func(temp_grid.grid[i].pos)
    end
    errors_at_particles = u_numerical .- u_analytical_at_particles
    abs_errors_at_particles = abs.(errors_at_particles)
    sq_errors_at_particles = errors_at_particles.^2

    # --- 3. Calculate Norms and Mass ---
    sum_l1_norm_val = 0.0
    sum_l2_norm_sq_val = 0.0
    max_sup_norm_val = 0.0
    sum_mass_val = 0.0
    
    res_interp = Vector{Float64}(undef, max(1, order + 1)) 
    # Assuming exponentialWeightFunction is defined, e.g., in Interpolations or FluxFunctions
    weight_func_default = Interpolations.exponentialWeightFunction() 

    for i in 1:N_particles
        particle_i = temp_grid.grid[i]
        vol_i = particle_i.volume

        # Pointwise values (used for supnorm and as fallback if no interpolation)
        val_num_i_pt = u_numerical[i] # u_numerical is directly from input argument
        val_abs_err_i_pt = abs_errors_at_particles[i]
        val_sq_err_i_pt = sq_errors_at_particles[i]

        # Values to be used in summation (default to pointwise)
        abs_err_for_sum = val_abs_err_i_pt
        sq_err_for_sum  = val_sq_err_i_pt
        u_num_for_sum   = val_num_i_pt
        
        num_neighbors_i = length(particle_i.neighbourIndices)

        # Perform interpolation if requested and feasible for 1D
        if order >= 0 && num_neighbors_i > 0 && num_neighbors_i >= order
            # Dynamically Sized Stencil Buffers for this particle
            dxVec_stencil = Vector{Float64}(undef, num_neighbors_i)
            wVec_stencil  = Vector{Float64}(undef, num_neighbors_i)
            # fVec_stencil will be populated per quantity
            
            for k_stencil in 1:num_neighbors_i
                nb_idx = particle_i.neighbourIndices[k_stencil]
                dxVec_stencil[k_stencil] = ParticleGrids.getPeriodicDistance(temp_grid, i, nb_idx)
            end
            
            # Calculate weights (wVec_stencil)
            # Assuming weight_func_default can write to wVec_stencil or returns a new vector
            try
                weight_func_default(dxVec_stencil; param=settings.interpAlpha, normalisation=temp_grid.dx, wVec_out=wVec_stencil)
            catch e
                if isa(e, MethodError)
                    wVec_stencil .= weight_func_default(dxVec_stencil; param=settings.interpAlpha, normalisation=temp_grid.dx)
                else rethrow(e) end
            end

            if "l1norm" in stats_to_calculate
                fVec_abs_err_stencil = Vector{Float64}(undef, num_neighbors_i)
                for k_stencil in 1:num_neighbors_i; fVec_abs_err_stencil[k_stencil] = abs_errors_at_particles[particle_i.neighbourIndices[k_stencil]]; end
                Interpolations.functionInterpolation!(dxVec_stencil, copy(wVec_stencil), fVec_abs_err_stencil, res_interp; order=order)
                abs_err_for_sum = res_interp[1]
            end
            if "l2norm" in stats_to_calculate
                fVec_sq_err_stencil = Vector{Float64}(undef, num_neighbors_i)
                for k_stencil in 1:num_neighbors_i; fVec_sq_err_stencil[k_stencil] = sq_errors_at_particles[particle_i.neighbourIndices[k_stencil]]; end
                Interpolations.functionInterpolation!(dxVec_stencil, copy(wVec_stencil), fVec_sq_err_stencil, res_interp; order=order)
                sq_err_for_sum = res_interp[1]
            end
            if "mass" in stats_to_calculate
                fVec_u_num_stencil = Vector{Float64}(undef, num_neighbors_i)
                for k_stencil in 1:num_neighbors_i; fVec_u_num_stencil[k_stencil] = u_numerical[particle_i.neighbourIndices[k_stencil]]; end
                Interpolations.functionInterpolation!(dxVec_stencil, copy(wVec_stencil), fVec_u_num_stencil, res_interp; order=order)
                u_num_for_sum = res_interp[1]
            end
        end # End if interp_order >= 0

        # Accumulate statistics
        if "l1norm" in stats_to_calculate
            sum_l1_norm_val += abs_err_for_sum * vol_i
        end
        if "l2norm" in stats_to_calculate
            sum_l2_norm_sq_val += sq_err_for_sum * vol_i 
        end
        if "supnorm" in stats_to_calculate
            if val_abs_err_i_pt > max_sup_norm_val # Sup norm uses pointwise error
                max_sup_norm_val = val_abs_err_i_pt
            end
        end
        if "mass" in stats_to_calculate
            sum_mass_val += u_num_for_sum * vol_i
        end
    end # End particle loop

    # Finalize results
    if "l1norm" in stats_to_calculate; results["l1norm"] = sum_l1_norm_val; end
    if "l2norm" in stats_to_calculate; results["l2norm"] = sqrt(sum_l2_norm_sq_val); end
    if "supnorm" in stats_to_calculate; results["supnorm"] = max_sup_norm_val; end
    if "mass" in stats_to_calculate; results["mass"] = sum_mass_val; end
    if "volume" in stats_to_calculate; results["volume"] = sum(p.volume for p in temp_grid.grid); end
    
    return results
end

function calculateAllStats!(
        sim_data::SimData1D,
        u_ana_func::Function,
        xmin::Real,
        xmax::Real,
        N::Int;
        settings::Union{SimSetting, Nothing} = nothing,
        order::Int = 2,
        stats_to_calculate::Vector{String} = ["l1norm", "l2norm", "supnorm", "mass"])

    for key = stats_to_calculate
        sim_data.stats[key] = []
    end
    for (m,t) = enumerate(sim_data.t)
        stats_tmp = calculateStats(sim_data.u[m], x -> u_ana_func(x, t), sim_data.x[m], xmin, xmax, N; settings = settings, order = order, stats_to_calculate = stats_to_calculate)
        for (key,val) = stats_tmp
            push!(sim_data.stats[key],val)
        end
    end

end

end # Module PlottingUtils