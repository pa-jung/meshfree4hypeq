module PlottingUtils

using ..ParticleGrids
using ..Particles
using ..SimSettings
#using ..Interpolations
using Dierckx
using QuadGK
using IPlotPDESols

export calculateStats, calculateAllStats!

"""
    calculate_stats_with_dierckx(
        u_numerical::AbstractVector{<:Real},
        u_analytical_func::Function, # Should take x (1D) or (x,y) (2D) -> analytical_value
        x_coords_input::Union{AbstractVector{<:Real}, AbstractVector{<:NTuple{2,Float64}}},
        domain_params::NamedTuple, # (xmin, xmax) for 1D; (xmin, xmax, ymin, ymax, Nx, Ny) for 2D
        N_particles::Int;
        dierckx_k::Int = 3, 
        dierckx_s::Union{Real,Nothing} = nothing, 
        quad_tol::Real = 1e-9, 
        stats_to_calculate::Vector{String} = ["l1norm", "l2norm", "supnorm", "mass"] # Removed "volume"
    ) -> Dict{String, Float64}

Calculates statistics using Dierckx.jl for spline interpolation and integration.
Supremum norm is calculated pointwise.
L1 norm for 1D uses QuadGK on abs(error_spline).
L2 norm for 1D uses QuadGK on (error_spline)^2.
Mass for 1D uses Dierckx.integrate on u_numerical_spline.
For 2D, L1/L2 norms are approximated by pointwise sums due to Dierckx.integrate limitations.
"""
function calculateStats(
    u_numerical::AbstractVector{<:Real},
    u_analytical_func::Function,
    x_coords_input::Union{AbstractVector{<:Real}, AbstractVector{<:NTuple{2,Float64}}},
    domain_params::NamedTuple,
    N_particles::Int;
    dierckx_k::Int = 3, 
    dierckx_s::Union{Real,Nothing} = nothing,
    quad_tol::Real = 1e-10,
    stats_to_calculate::Vector{String} = ["l1norm", "l2norm", "supnorm", "mass"] # "volume" removed
)::Dict{String, Float64}

    if N_particles == 0
        @warn "Numerical solution vector is empty. Returning empty stats."
        return Dict{String, Float64}()
    end
    if length(u_numerical) != N_particles || length(x_coords_input) != N_particles
        error("Input u_numerical and x_coords_input lengths must match N_particles.")
    end

    results = Dict{String, Float64}()
    
    s_val_actual = isnothing(dierckx_s) ? Float64(N_particles) - sqrt(2.0*Float64(N_particles)) : Float64(dierckx_s)
    if s_val_actual < 0.0; s_val_actual = 0.0; end

    u_analytical_at_particles = Vector{Float64}(undef, N_particles)
    if eltype(x_coords_input) <: Real # 1D
        for i in 1:N_particles
            u_analytical_at_particles[i] = u_analytical_func(x_coords_input[i])
        end
    else # 2D (NTuple{2,Float64})
        for i in 1:N_particles
            u_analytical_at_particles[i] = u_analytical_func(x_coords_input[i]...)
        end
    end
    errors_at_particles = u_numerical .- u_analytical_at_particles

    if "supnorm" in stats_to_calculate
        results["supnorm"] = N_particles > 0 ? maximum(abs.(errors_at_particles)) : 0.0
    end

    if eltype(x_coords_input) <: Real # 1D Case
        xmin, xmax = domain_params.xmin, domain_params.xmax
        
        perm = sortperm(x_coords_input)
        x_sorted = x_coords_input[perm]
        u_num_sorted = u_numerical[perm]
        errors_sorted = errors_at_particles[perm]

        spl_u_num = Dierckx.Spline1D(x_sorted, u_num_sorted; k=dierckx_k, s=s_val_actual, bc="nearest")
        spl_error = Dierckx.Spline1D(x_sorted, errors_sorted; k=dierckx_k, s=s_val_actual, bc="nearest")

        if "mass" in stats_to_calculate
            results["mass"] = Dierckx.integrate(spl_u_num, xmin, xmax)
        end
        if "l1norm" in stats_to_calculate
            l1_val, _ = QuadGK.quadgk(x -> abs(spl_error(x)), xmin, xmax, rtol=quad_tol, atol=quad_tol^2)
            results["l1norm"] = l1_val
        end
        if "l2norm" in stats_to_calculate
            l2_sq_val, _ = QuadGK.quadgk(x -> spl_error(x)^2, xmin, xmax, rtol=quad_tol, atol=quad_tol^2)
            results["l2norm"] = sqrt(l2_sq_val)
        end

    elseif eltype(x_coords_input) <: NTuple{2,Float64} # 2D Case
        xmin, xmax = domain_params.xmin, domain_params.xmax
        ymin, ymax = domain_params.ymin, domain_params.ymax

        x_vec_2d = [pt[1] for pt in x_coords_input]
        y_vec_2d = [pt[2] for pt in x_coords_input]

        spl_u_num_2D = Dierckx.Spline2D(x_vec_2d, y_vec_2d, u_numerical; kx=dierckx_k, ky=dierckx_k, s=s_val_actual)
        # spl_error_2D = Dierckx.Spline2D(x_vec_2d, y_vec_2d, errors_at_particles; kx=dierckx_k, ky=dierckx_k, s=s_val_actual)

        if "mass" in stats_to_calculate
            results["mass"] = Dierckx.integrate(spl_u_num_2D, xmin, xmax, ymin, ymax)
        end
        
        # For 2D L1 and L2, Dierckx.integrate integrates the spline, not abs(spline) or spline^2.
        # Using pointwise sum with volumes as an approximation if HCubature/Cuba not available.
        if "l1norm" in stats_to_calculate || "l2norm" in stats_to_calculate
            @warn "For 2D, L1 and L2 norms are approximated by pointwise summation using particle volumes. For higher accuracy, consider a 2D quadrature package."
            
            # Create temporary ParticleGrid2D to get volumes
            temp_particles_2D = [Meshfree4ScalarEq.Particles.Particle2D(x_coords_input[i], u_numerical[i], false) for i in 1:N_particles]
            temp_grid_2D_obj = Meshfree4ScalarEq.ParticleGrids.ParticleGrid2D(
                domain_params.xmin, domain_params.xmax, domain_params.ymin, domain_params.ymax,
                get(domain_params, :Nx, round(Int,sqrt(N_particles))), # Get Nx, Ny from domain_params
                get(domain_params, :Ny, round(Int,sqrt(N_particles))); 
                randomness = (0.0, 0.0) 
            )
            if length(temp_grid_2D_obj.grid) == N_total_particles
                for i_pg in 1:N_total_particles; temp_grid_2D_obj.grid[i_pg].pos = x_coords_input[i_pg]; end
                determineVolumes_placeholder!(temp_grid_2D_obj) # Your function
            else
                error("Temp grid particle count mismatch for 2D volume calculation in stats.")
            end

            sum_l1_val_2D = 0.0
            sum_l2_sq_val_2D = 0.0
            for i in 1:N_particles
                vol_i = temp_grid_2D_obj.grid[i].volume
                # Error is already calculated pointwise in errors_at_particles
                error_val_at_particle = errors_at_particles[i] 

                if "l1norm" in stats_to_calculate
                    sum_l1_val_2D += abs(error_val_at_particle) * vol_i
                end
                if "l2norm" in stats_to_calculate
                    sum_l2_sq_val_2D += error_val_at_particle^2 * vol_i
                end
            end
            if "l1norm" in stats_to_calculate; results["l1norm"] = sum_l1_val_2D; end
            if "l2norm" in stats_to_calculate; results["l2norm"] = sqrt(sum_l2_sq_val_2D); end
        end
    else
        error("Unsupported x_coords_input element type: $(eltype(x_coords_input))")
    end
    
    return results
end

function calculateAllStats!(
        sim_data::SimData1D,
        u_ana_func::Function,
        domain_params::NamedTuple,
        N::Int;
        dierckx_k::Int = 3, 
        dierckx_s::Union{Real,Nothing} = nothing,
        quad_tol::Real = 1e-9,
        stats_to_calculate::Vector{String} = ["l1norm", "l2norm", "supnorm", "mass"] # "volume" removed
    )
    for key = stats_to_calculate
        sim_data.stats[key] = []
    end
    for (m,t) = enumerate(sim_data.t)
        stats_tmp = calculateStats(sim_data.u[m], 
                                   x -> u_ana_func(x, t), 
                                   sim_data.x[m], 
                                   domain_params, N; 
                                   quad_tol = quad_tol, 
                                   dierckx_k = dierckx_k, 
                                   stats_to_calculate = stats_to_calculate,
                                   dierckx_s = dierckx_s)
        for (key,val) = stats_tmp
            push!(sim_data.stats[key],val)
        end
    end

end
# # Uses Meshfree interpolation
# function calculateStatsMeshfree(
#     u_numerical::AbstractVector{<:Real},
#     u_analytical_func::Function,
#     x_coords::AbstractVector{<:Real},
#     xmin_domain::Real,
#     xmax_domain::Real,
#     N_particles::Int; # Make sure this is consistent with lengths of x_coords and u_numerical
#     settings::SimSettings.SimSetting,
#     order::Int = 0, 
#     stats_to_calculate::Vector{String} = ["l1norm", "l2norm", "supnorm", "mass", "volume"]
# )::Dict{String, Float64}

#     if N_particles == 0
#         @warn "Numerical solution vector is empty based on N_particles. Returning empty stats."
#         return Dict{String, Float64}()
#     end
#     if length(u_numerical) != N_particles || length(x_coords) != N_particles
#         error("Input u_numerical and x_coords lengths ($(length(u_numerical)), $(length(x_coords))) must match N_particles ($N_particles).")
#     end

#     results = Dict{String, Float64}()

#     # --- 1. Create and Initialize Temporary ParticleGrid1D ---
#     temp_particles = [Particles.Particle1D(x_coords[i], u_numerical[i], false) for i in 1:N_particles]
#     if !isempty(temp_particles)
#         temp_particles[1].boundary = true # As per your ParticleGrid1D constructor constraint
#     end
    
#     nominal_dx = (xmax_domain - xmin_domain) / (N_particles > 0 ? N_particles : 1)
#     is_regular_grid = true # Determine this robustly if necessary for your ParticleGrid constructor
#     if N_particles > 1
#         for i_reg in 2:N_particles
#             if !isapprox(x_coords[i_reg] - x_coords[i_reg-1], nominal_dx, atol=1e-9 * abs(nominal_dx))
#                 is_regular_grid = false; break
#             end
#         end
#     end
    
#     # Use the ParticleGrid1D constructor that takes a vector of particles and other geometric info
#     temp_grid = ParticleGrids.ParticleGrid1D(temp_particles, xmin_domain, xmax_domain, nominal_dx, is_regular_grid)
    
#     determineVolumes!(temp_grid) # YOU PROVIDE THIS FUNCTION - it populates particle.volume
#     ParticleGrids.updateNeighbours!(temp_grid, 5.) #Ensure enough neighbors #settings.interpRange) # Populates particle.neighbourIndices

#     # --- 2. Calculate Analytical Solution and Errors at particle locations ---
#     u_analytical_at_particles = Vector{Float64}(undef, N_particles)
#     for i in 1:N_particles
#          # Assuming u_analytical_func takes only x for 1D
#         u_analytical_at_particles[i] = u_analytical_func(temp_grid.grid[i].pos)
#     end
#     errors_at_particles = u_numerical .- u_analytical_at_particles
#     abs_errors_at_particles = abs.(errors_at_particles)
#     sq_errors_at_particles = errors_at_particles.^2

#     # --- 3. Calculate Norms and Mass ---
#     sum_l1_norm_val = 0.0
#     sum_l2_norm_sq_val = 0.0
#     max_sup_norm_val = 0.0
#     sum_mass_val = 0.0
    
#     res_interp = Vector{Float64}(undef, max(1, order + 1)) 
#     # Assuming exponentialWeightFunction is defined, e.g., in Interpolations or FluxFunctions
#     weight_func_default = Interpolations.exponentialWeightFunction() 

#     for i in 1:N_particles
#         particle_i = temp_grid.grid[i]
#         vol_i = particle_i.volume

#         # Pointwise values (used for supnorm and as fallback if no interpolation)
#         val_num_i_pt = u_numerical[i] # u_numerical is directly from input argument
#         val_abs_err_i_pt = abs_errors_at_particles[i]
#         val_sq_err_i_pt = sq_errors_at_particles[i]

#         # Values to be used in summation (default to pointwise)
#         abs_err_for_sum = val_abs_err_i_pt
#         sq_err_for_sum  = val_sq_err_i_pt
#         u_num_for_sum   = val_num_i_pt
        
#         num_neighbors_i = length(particle_i.neighbourIndices)

#         # Perform interpolation if requested and feasible for 1D
#         if order >= 0 && num_neighbors_i > 0 && num_neighbors_i >= order
#             # Dynamically Sized Stencil Buffers for this particle
#             dxVec_stencil = Vector{Float64}(undef, num_neighbors_i)
#             wVec_stencil  = Vector{Float64}(undef, num_neighbors_i)
#             # fVec_stencil will be populated per quantity
            
#             for k_stencil in 1:num_neighbors_i
#                 nb_idx = particle_i.neighbourIndices[k_stencil]
#                 dxVec_stencil[k_stencil] = ParticleGrids.getPeriodicDistance(temp_grid, i, nb_idx)
#             end
            
#             # Calculate weights (wVec_stencil)
#             # Assuming weight_func_default can write to wVec_stencil or returns a new vector
#             try
#                 weight_func_default(dxVec_stencil; param=settings.interpAlpha, normalisation=temp_grid.dx, wVec_out=wVec_stencil)
#             catch e
#                 if isa(e, MethodError)
#                     wVec_stencil .= weight_func_default(dxVec_stencil; param=settings.interpAlpha, normalisation=temp_grid.dx)
#                 else rethrow(e) end
#             end

#             if "l1norm" in stats_to_calculate
#                 fVec_abs_err_stencil = Vector{Float64}(undef, num_neighbors_i)
#                 for k_stencil in 1:num_neighbors_i; fVec_abs_err_stencil[k_stencil] = abs_errors_at_particles[particle_i.neighbourIndices[k_stencil]]; end
#                 Interpolations.functionInterpolation!(dxVec_stencil, copy(wVec_stencil), fVec_abs_err_stencil, res_interp; order=order)
#                 abs_err_for_sum = res_interp[1]
#             end
#             if "l2norm" in stats_to_calculate
#                 fVec_sq_err_stencil = Vector{Float64}(undef, num_neighbors_i)
#                 for k_stencil in 1:num_neighbors_i; fVec_sq_err_stencil[k_stencil] = sq_errors_at_particles[particle_i.neighbourIndices[k_stencil]]; end
#                 Interpolations.functionInterpolation!(dxVec_stencil, copy(wVec_stencil), fVec_sq_err_stencil, res_interp; order=order)
#                 sq_err_for_sum = res_interp[1]
#             end
#             if "mass" in stats_to_calculate
#                 fVec_u_num_stencil = Vector{Float64}(undef, num_neighbors_i)
#                 for k_stencil in 1:num_neighbors_i; fVec_u_num_stencil[k_stencil] = u_numerical[particle_i.neighbourIndices[k_stencil]]; end
#                 Interpolations.functionInterpolation!(dxVec_stencil, copy(wVec_stencil), fVec_u_num_stencil, res_interp; order=order)
#                 u_num_for_sum = res_interp[1]
#             end
#         end # End if interp_order >= 0

#         # Accumulate statistics
#         if "l1norm" in stats_to_calculate
#             sum_l1_norm_val += abs_err_for_sum * vol_i
#         end
#         if "l2norm" in stats_to_calculate
#             sum_l2_norm_sq_val += sq_err_for_sum * vol_i 
#         end
#         if "supnorm" in stats_to_calculate
#             if val_abs_err_i_pt > max_sup_norm_val # Sup norm uses pointwise error
#                 max_sup_norm_val = val_abs_err_i_pt
#             end
#         end
#         if "mass" in stats_to_calculate
#             sum_mass_val += u_num_for_sum * vol_i
#         end
#     end # End particle loop

#     # Finalize results
#     if "l1norm" in stats_to_calculate; results["l1norm"] = sum_l1_norm_val; end
#     if "l2norm" in stats_to_calculate; results["l2norm"] = sqrt(sum_l2_norm_sq_val); end
#     if "supnorm" in stats_to_calculate; results["supnorm"] = max_sup_norm_val; end
#     if "mass" in stats_to_calculate; results["mass"] = sum_mass_val; end
#     if "volume" in stats_to_calculate; results["volume"] = sum(p.volume for p in temp_grid.grid); end
    
#     return results
# end



end # Module PlottingUtils