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
    calculateStats(
        u_numerical::AbstractVector{<:Real},
        u_analytical_func::Function,
        x_coords_input::Union{AbstractVector{<:Real}, AbstractVector{<:NTuple{2,Float64}}},
        domain_params::NamedTuple,
        N_particles::Int;
        ...
    ) -> Dict{String, Float64}

Calculates normalized statistics for a numerical solution by comparing it
to an analytical solution. Uses Dierckx.jl and QuadGK.jl for accurate
interpolation and integration to compute relative L1, L2, and sup norms,
as well as relative mass and wave position error.
"""
function calculateStats(
    u_numerical::AbstractVector{<:Real},
    u_analytical_func::Function,
    x_coords_input::Union{AbstractVector{<:Real}, AbstractVector{<:NTuple{2,Float64}}},
    domain_params::NamedTuple,
    N_particles::Int;
    dierckx_k::Int = 3, 
    dierckx_s::Union{Real,Nothing} = nothing,
    quad_tol::Real = 1e-10, # Increased tolerance slightly for stability
    stats_to_calculate::Vector{String} = [
        "l1norm", "l2norm", "supnorm", "mass", "relative_mass",
        "wave_position_error", "wave_height_error",
        "discrete_l2norm", "discrete_l1norm"
    ]
)::Dict{String, Float64}

    if N_particles == 0
        @warn "Numerical solution vector is empty. Returning empty stats."
        return Dict{String, Float64}()
    end
    if length(u_numerical) != N_particles || length(x_coords_input) != N_particles
        error("Input u_numerical and x_coords_input lengths must match N_particles.")
    end

    results = Dict{String, Float64}()
    
    s_val_actual = isnothing(dierckx_s) ? max(0.0, Float64(N_particles) - sqrt(2.0*Float64(N_particles))) : Float64(dierckx_s)
    if s_val_actual < 0.0; s_val_actual = 0.0; end

    # --- 1. Calculate Analytical Solution and Pointwise Errors ---
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

    # --- 2. Supremum Norm (Pointwise) ---
    if "supnorm" in stats_to_calculate || "supnorm" in stats_to_calculate
        # Sup norm of the error
        sup_norm_error = N_particles > 0 ? maximum(abs.(errors_at_particles)) : 0.0
        results["supnorm"] = sup_norm_error # Store absolute sup norm
        
        # Sup norm of the analytical solution for normalization
        sup_norm_analytical = N_particles > 0 ? maximum(abs.(u_analytical_at_particles)) : 0.0
        if sup_norm_analytical > 1e-12
            results["supnorm"] = sup_norm_error / sup_norm_analytical
        else
            results["supnorm"] = sup_norm_error # Avoid division by zero, return absolute error
        end
    end

    # --- 3. Discrete, Unweighted Vector Norms ---
    if any(s -> occursin("discrete", s), stats_to_calculate)
        # Discrete L1 error norm: sum of absolute errors
        discrete_l1_error = sum(abs.(errors_at_particles))
        results["discrete_l1norm"] = discrete_l1_error
        
        # Discrete L2 error norm: sqrt of sum of squared errors
        # Using LinearAlgebra.norm is concise and efficient for this.
        # using LinearAlgebra; discrete_l2_error = norm(errors_at_particles, 2)
        discrete_l2_error = sqrt(sum(errors_at_particles.^2))
        results["discrete_l2norm"] = discrete_l2_error

        # Calculate analytical norms for normalization
        discrete_l1_analytical = sum(abs.(u_analytical_at_particles))
        discrete_l2_analytical = sqrt(sum(u_analytical_at_particles.^2))

        if "discrete_l1norm" in stats_to_calculate
            results["discrete_l1norm"] = discrete_l1_analytical > 1e-12 ? discrete_l1_error / discrete_l1_analytical : discrete_l1_error
        end
        if "discrete_l2norm" in stats_to_calculate
            results["discrete_l2norm"] = discrete_l2_analytical > 1e-12 ? discrete_l2_error / discrete_l2_analytical : discrete_l2_error
        end
    end

    # --- 3. Interpolation and Integration for Norms and Mass ---
    if eltype(x_coords_input) <: Real # 1D Case
        xmin, xmax = domain_params.xmin, domain_params.xmax
        
        local x_sorted, u_num_sorted, errors_sorted, u_ana_sorted
        if N_particles > 1 && !issorted(x_coords_input)
            perm = sortperm(x_coords_input)
            x_sorted = x_coords_input[perm]
            u_num_sorted = u_numerical[perm]
            errors_sorted = errors_at_particles[perm]
            u_ana_sorted = u_analytical_at_particles[perm]
        else
            x_sorted = x_coords_input
            u_num_sorted = u_numerical
            errors_sorted = errors_at_particles
            u_ana_sorted = u_analytical_at_particles
        end

        # --- Mass Calculation (Spline vs Spline) ---
        if ("mass" in stats_to_calculate || "relative_mass" in stats_to_calculate) && N_particles > 0
            mass_num, mass_ana = NaN, NaN
            try
                spl_u_num = Dierckx.Spline1D(x_sorted, u_num_sorted; k=dierckx_k, s=s_val_actual, bc="nearest")
                mass_num = Dierckx.integrate(spl_u_num, xmin, xmax)
                
                spl_u_ana = Dierckx.Spline1D(x_sorted, u_ana_sorted; k=dierckx_k, s=s_val_actual, bc="nearest")
                mass_ana = Dierckx.integrate(spl_u_ana, xmin, xmax)
            catch e
                @warn "Dierckx.Spline1D for mass calculation failed: $e. Mass not computed."
            end
            
            if "mass" in stats_to_calculate; results["mass"] = mass_num; end
            if "relative_mass" in stats_to_calculate
                results["relative_mass"] = (abs(mass_ana) > 1e-12 && !isnan(mass_num)) ? mass_num / mass_ana : NaN
            end
        end
        
        # --- Integrated Error Norms (Error Spline vs. Analytical Function Integral) ---
        if ("l1norm" in stats_to_calculate || "l2norm" in stats_to_calculate) && N_particles > 1
            spl_error_1D = nothing
            try spl_error_1D = Dierckx.Spline1D(x_sorted, errors_sorted; k=dierckx_k, s=s_val_actual, bc="nearest")
            catch e; @warn "Dierckx.Spline1D for error failed: $e. Integrated L1/L2 norms will be NaN."; end

            if !isnothing(spl_error_1D)
                # Normalization denominator is the norm of the TRUE analytical function
                ana_l1_norm_integrated, _ = QuadGK.quadgk(x -> abs(u_analytical_func(x)), xmin, xmax, rtol=quad_tol)
                ana_l2_norm_sq_integrated, _ = QuadGK.quadgk(x -> u_analytical_func(x)^2, xmin, xmax, rtol=quad_tol)
                ana_l2_norm_integrated = sqrt(ana_l2_norm_sq_integrated)

                if "l1norm" in stats_to_calculate
                    l1_err_val, _ = QuadGK.quadgk(x -> abs(spl_error_1D(x)), xmin, xmax, rtol=quad_tol)
                    results["l1norm"] = ana_l1_norm_integrated > 1e-12 ? l1_err_val / ana_l1_norm_integrated : l1_err_val
                end
                if "l2norm" in stats_to_calculate
                    l2_err_sq_val, _ = QuadGK.quadgk(x -> spl_error_1D(x)^2, xmin, xmax, rtol=quad_tol)
                    results["l2norm"] = ana_l2_norm_integrated > 1e-12 ? sqrt(l2_err_sq_val) / ana_l2_norm_integrated : sqrt(l2_err_sq_val)
                end
            else # spl_error_1D failed to be created
                 if "l1norm" in stats_to_calculate; results["l1norm"] = NaN; end
                 if "l2norm" in stats_to_calculate; results["l2norm"] = NaN; end
            end
        end
    elseif eltype(x_coords_input) <: NTuple{2,Float64} # 2D Case
        @warn "Integrated L1/L2 norms for 2D are not yet implemented with a high-order quadrature package. Discrete norms can be used as a proxy."
        # If stats_to_calculate contains "l1norm" or "l2norm", report NaN as they are not computed.
        if "l1norm" in stats_to_calculate; results["l1norm"] = NaN; end
        if "l2norm" in stats_to_calculate; results["l2norm"] = NaN; end
        # Mass calculation for 2D via Dierckx can still be done
        if "mass" in stats_to_calculate && N_particles > 0
            xmin, xmax = domain_params.xmin, domain_params.xmax
            ymin, ymax = domain_params.ymin, domain_params.ymax
            x_vec_2d = [pt[1] for pt in x_coords_input]
            y_vec_2d = [pt[2] for pt in x_coords_input]
            try
                spl_u_num_2D = Dierckx.Spline2D(x_vec_2d, y_vec_2d, u_numerical; kx=dierckx_k, ky=dierckx_k, s=s_val_actual)
                results["mass"] = Dierckx.integrate(spl_u_num_2D, xmin, xmax, ymin, ymax)
            catch e
                @warn "Dierckx.Spline2D for u_numerical (mass) failed: $e. Mass not computed."
                results["mass"] = NaN
            end
        elseif "mass" in stats_to_calculate
            results["mass"] = 0.0
        end
    end

    # --- Wave Tracking (Pointwise) ---
    if ("wave_position_error" in stats_to_calculate || "wave_height_error" in stats_to_calculate) && N_particles > 0
        height_ana, index_ana = findmax(u_analytical_at_particles)
        pos_ana = x_coords_input[index_ana]
        height_num, index_num = findmax(u_numerical)
        pos_num = x_coords_input[index_num]
        
        if "wave_position_error" in stats_to_calculate
            domain_length = eltype(x_coords_input) <: Real ? (domain_params.xmax - domain_params.xmin) : 1.0 # Placeholder for 2D domain size
            results["wave_position_error"] = domain_length > 1e-9 ? abs(pos_ana - pos_num) / domain_length : abs(pos_ana - pos_num)
        end
        if "wave_height_error" in stats_to_calculate
            results["wave_height_error"] = abs(height_ana) > 1e-9 ? abs(height_ana - height_num) / abs(height_ana) : abs(height_ana - height_num)
        end
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
        stats_to_calculate::Vector{String} = [
        "l1norm", "l2norm", "supnorm", "mass", "relative_mass",
        "discrete_l1norm", "discrete_l2norm", "wave_position_error", "wave_height_error"
    ]
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