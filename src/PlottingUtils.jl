module PlottingUtils

using ..ParticleGrids
using ..Particles
using ..SimSettings
using ..InitialConditions # For InitialCondition and get_discontinuity_points
using ..ScalarHyperbolicEquations # For equation types
#using ..Interpolations
using Dierckx
using QuadGK
using IPlotPDESols


export calculateStats, calculateAllStats!


"""
    _augment_data_for_spline(x_coords, y_values, discontinuity_points)

Augments a dataset with "double points" at known discontinuities to
ensure correct spline interpolation of jumps. This version uses a robust
manual search for neighbor values instead of creating an intermediate spline.
"""
function _augment_data_for_spline(
    x_coords::AbstractVector{<:Real}, 
    y_values::AbstractVector{<:Real}, 
    discontinuity_points::AbstractVector{<:Real}
)
    if isempty(discontinuity_points) || isempty(x_coords)
        return copy(x_coords), copy(y_values)
    end

    x_new = copy(x_coords)
    y_new = copy(y_values)

    for xd in discontinuity_points
        # Find the index of the last point to the left of the discontinuity
        idx_left = findlast(x -> x < xd, x_coords)
        # Find the index of the first point to the right of the discontinuity
        idx_right = findfirst(x -> x > xd, x_coords)

        # Determine the values on the left and right of the jump
        # by using the nearest available data point.
        val_left = isnothing(idx_left) ? y_values[idx_right] : y_values[idx_left]
        val_right = isnothing(idx_right) ? y_values[idx_left] : y_values[idx_right]

        # Add two points at the discontinuity: one for the left state, one for the right
        # This tells Dierckx to create a jump in the spline.
        push!(x_new, xd)
        push!(y_new, val_left)
        
        push!(x_new, xd)
        push!(y_new, val_right)
    end
    
    # Return the augmented data, sorted by x-coordinate for Dierckx
    p_new = sortperm(x_new)
    return x_new[p_new], y_new[p_new]
end
"""
    _create_piecewise_spline_function(x_coords, y_values, domain_params, discontinuity_points, k)

Creates a callable, piecewise spline function. The domain is broken into smooth
sub-intervals at the discontinuity points. A separate spline is created for each piece.
"""
function _create_piecewise_spline_function(
    x_coords::AbstractVector{<:Real}, 
    y_values::AbstractVector{<:Real}, 
    breakpoints::AbstractVector{<:Real},
    dierckx_k::Int
)
    if isempty(x_coords)
        return x -> 0.0 # Return a zero function if there's no data
    end

    splines = Dierckx.Spline1D[] # A vector to hold a spline for each smooth segment

    for i in 1:(length(breakpoints)-1)
        # Define the current smooth sub-interval
        xa = breakpoints[i]
        xb = breakpoints[i+1]

        # Find all data points within this sub-interval
        # Add a small epsilon to include points exactly at the boundaries
        epsilon = 1e-9
        indices_in_sub = findall(x -> (xa - epsilon) <= x <= (xb + epsilon), x_coords)

        if length(indices_in_sub) < dierckx_k + 1
            # Not enough points for the requested spline order, fallback to linear
            current_k = 1
            if length(indices_in_sub) < 2
                # Not enough points even for linear, use constant from nearest point
                if isempty(indices_in_sub)
                    # If no points in interval, find nearest point overall to create a constant spline
                    _, nearest_idx = findmin(val -> abs(val - (xa+xb)/2), x_coords)
                    push!(splines, Dierckx.Spline1D([xa, xb], [y_values[nearest_idx], y_values[nearest_idx]]; k=1, bc="nearest"))
                else
                    # Only one point, create a constant spline
                    push!(splines, Dierckx.Spline1D([xa, xb], [y_values[indices_in_sub[1]], y_values[indices_in_sub[1]]]; k=1, bc="nearest"))
                end
                continue
            end
        else
            current_k = dierckx_k
        end
        
        # Get the data for this piece
        x_piece = x_coords[indices_in_sub]
        y_piece = y_values[indices_in_sub]

        # To ensure the spline is well-defined at the boundaries of the sub-interval,
        # we can add the boundary points themselves using constant interpolation.
        # This uses the value of the closest data point as the value at the boundary.
        if abs(x_piece[1] - xa) > epsilon
            insert!(x_piece, 1, xa)
            insert!(y_piece, 1, y_piece[1]) # Constant extrapolation
        end
        if abs(x_piece[end] - xb) > epsilon
            push!(x_piece, xb)
            push!(y_piece, y_piece[end]) # Constant extrapolation
        end

        try
            # Create a spline for this smooth piece of the domain
            spl = Dierckx.Spline1D(x_piece, y_piece; k=current_k, s=0., bc="nearest")
            push!(splines, spl)
        catch e
            @warn "Dierckx spline creation failed for sub-interval [$xa, $xb]: $e. Adding a zero-spline."
            # Add a placeholder spline that evaluates to zero
            push!(splines, Dierckx.Spline1D([xa, xb], [0.0, 0.0]; k=1))
        end
    end

    # Return a function that evaluates the correct spline based on x
    return function piecewise_spline(x::Real)
        # Find which sub-interval x falls into
        # `searchsortedlast` finds the index of the last breakpoint <= x
        idx = searchsortedlast(breakpoints, x)
        
        # Handle edges
        if idx == 0; idx = 1; end
        if idx >= length(breakpoints); idx = length(splines); end
        
        return splines[idx](x)
    end
end
"""
    _calculate_stats_at_timestep(...)

Internal helper function to compute statistics for a single time step.
"""
function _calculate_stats_at_timestep(
    u_numerical::AbstractVector{<:Real},
    x_coords::AbstractVector{<:Real},
    t::Real,
    pg::ParticleGrid1D,
    ic::InitialCondition,
    eq::ScalarHyperbolicEquation;
    dierckx_k::Int,
    quad_tol::Real
)::Dict{String, Float64}

    results = Dict{String, Float64}()
    N_particles = length(u_numerical)
    if N_particles == 0; return results; end

    # Define the analytical function closure for this specific time t
    analytical_func_at_t = x -> ic(x, t, eq, pg)

    # --- 1. Calculate Pointwise and Analytical Values ---
    u_analytical_at_particles = [analytical_func_at_t(x) for x in x_coords]
    errors_at_particles = u_numerical .- u_analytical_at_particles

    # Define domain parameters
    xmin, xmax = pg.xmin, pg.xmax
    domain_length = xmax - xmin
    
    # Get discontinuity points for QuadGK at the current time t
    discontinuity_points = get_discontinuity_points(ic, eq, t, pg)
    breakpoints = unique([xmin; discontinuity_points; xmax])

    # Insert discontinuity points
    x_coords_aug, u_aug_num = _augment_data_for_spline(x_coords, u_numerical, discontinuity_points)
    _, err_aug = _augment_data_for_spline(x_coords, errors_at_particles, discontinuity_points)
    # --- 2. Calculate High-Accuracy Analytical Norms/Mass via QuadGK ---
    ana_l1_norm, _ = QuadGK.quadgk(x -> abs(analytical_func_at_t(x)), breakpoints...; rtol=quad_tol)
    ana_l2_sq_norm, _ = QuadGK.quadgk(x -> analytical_func_at_t(x)^2, breakpoints...; rtol=quad_tol)
    ana_l2_norm = sqrt(ana_l2_sq_norm)
    mass_ana, _ = QuadGK.quadgk(analytical_func_at_t, breakpoints...; rtol=quad_tol)

    # --- 3. Create Splines from Discrete Data ---
    perm = sortperm(x_coords)
    x_sorted = x_coords[perm]
    spl_error = _create_piecewise_spline_function(x_sorted, errors_at_particles[perm], breakpoints, dierckx_k)#
    #spl_error = Dierckx.Spline1D(x_coords_aug, err_aug; k=dierckx_k, s=0.0, bc="nearest")
    spl_u_num = _create_piecewise_spline_function(x_sorted, u_numerical[perm], breakpoints, dierckx_k)#
    #spl_u_num = Dierckx.Spline1D(x_coords_aug, u_aug_num; k=dierckx_k, s=0.0, bc="nearest")

    # --- 4. Calculate All Requested Statistics using Splines and Analytical Norms ---
    
    # Integrated Error Norms (from error spline)
    l1_error_val, _ = QuadGK.quadgk(x -> abs(spl_error(x)), breakpoints...; rtol=quad_tol)
    l2_sq_error_val, _ = QuadGK.quadgk(x -> spl_error(x)^2, breakpoints...; rtol=quad_tol)
    results["l1error"] = l1_error_val
    results["l2error"] = sqrt(l2_sq_error_val)
    
    # Relative Integrated Error Norms
    results["relative_l1error"] = ana_l1_norm > 1e-12 ? results["l1error"] / ana_l1_norm : results["l1error"]
    results["relative_l2error"] = ana_l2_norm > 1e-12 ? results["l2error"] / ana_l2_norm : results["l2error"]

    # Integrated Solution Norms (from numerical spline)
    l1_norm_val, _ = QuadGK.quadgk(x -> abs(spl_u_num(x)), breakpoints...; rtol=quad_tol)
    l2_sq_norm_val, _ = QuadGK.quadgk(x -> spl_u_num(x)^2, breakpoints...; rtol=quad_tol)
    results["l1norm"] = l1_norm_val
    results["l2norm"] = sqrt(l2_sq_norm_val)

    # Mass
    mass_num,_ = QuadGK.quadgk(x -> spl_u_num(x), breakpoints...; rtol=quad_tol)
    results["mass"] = mass_num
    results["relative_mass"] = abs(mass_ana) > 1e-12 ? mass_num / abs(mass_ana) : NaN
    
    # Supremum Norm (pointwise)
    results["supnorm"] = maximum(abs.(errors_at_particles))
    sup_norm_ana = maximum(abs.(u_analytical_at_particles))
    results["relative_supnorm"] = sup_norm_ana > 1e-12 ? results["supnorm"] / sup_norm_ana : results["supnorm"]

    height_ana, index_ana = findmax(u_analytical_at_particles)
    pos_ana = x_coords[index_ana]
    height_num, index_num = findmax(u_numerical)
    pos_num = x_coords[index_num]

    results["wave_position_error"] = domain_length > 1e-9 ? abs(pos_ana - pos_num) / domain_length : abs(pos_ana - pos_num)
    results["wave_height_error"] = abs(height_ana) > 1e-9 ? abs(height_ana - height_num) / abs(height_ana) : abs(height_ana - height_num)

    return results
end


"""
    calculateAllStats!(sim_data::AbstractSimData, ic_object::InitialCondition, eq::ScalarHyperbolicEquation; ...)

Main user-facing function. Loops through all time steps in `sim_data`,
calculates a comprehensive set of statistics for each step, and stores
them in `sim_data.stats`.
"""
function calculateAllStats!(
    sim_data::AbstractSimData,
    ic_object::InitialCondition,
    eq::ScalarHyperbolicEquation,
    pg::ParticleGrid;
    dierckx_k::Int = 3, 
    quad_tol::Real = 1e-12,
    stats_to_calculate::Union{String,Vector{String}} = "all"
)
    # Define the full list of possible stats
    all_possible_stats = [
        "l1error", "l2error", "supnorm", 
        "relative_l1error", "relative_l2error", "relative_supnorm",
        "l1norm", "l2norm", "mass", "relative_mass", "wave_position_error",
        "wave_height_error"
    ]
    
    stats_list = stats_to_calculate == "all" ? all_possible_stats : stats_to_calculate

    # Initialize stats dictionary
    if !hasproperty(sim_data, :stats) || !isa(sim_data.stats, Dict)
        sim_data.stats = Dict{String, Any}()
    end
    for key in stats_list
        sim_data.stats[key] = [] # Initialize as empty vector
    end
    for (m, t) in enumerate(sim_data.t)
        # For each time step, we need a particle grid object to pass to the analytical solution
        # This grid contains the positions and boundary condition info for that time step.
        x_coords = sim_data.x[m]

        # Call the helper function for this time step
        stats_tmp = _calculate_stats_at_timestep(
            sim_data.u[m],
            x_coords,
            t,
            pg,
            ic_object,
            eq;
            dierckx_k = dierckx_k,
            quad_tol = quad_tol
        )
        
        # Append results
        for key in stats_list
            if haskey(stats_tmp, key)
                push!(sim_data.stats[key], stats_tmp[key])
            else
                # Push NaN if a stat wasn't calculated (e.g., due to error)
                push!(sim_data.stats[key], NaN)
            end
        end
    end
end

# """
#     calculateStats(
#         u_numerical::AbstractVector{<:Real},
#         u_analytical_func::Function,
#         x_coords_input::Union{AbstractVector{<:Real}, AbstractVector{<:NTuple{2,Float64}}},
#         domain_params::NamedTuple,
#         N_particles::Int;
#         ...
#     ) -> Dict{String, Float64}

# Calculates normalized statistics for a numerical solution by comparing it
# to an analytical solution. Uses Dierckx.jl and QuadGK.jl for accurate
# interpolation and integration to compute relative L1, L2, and sup norms,
# as well as relative mass and wave position error.
# """
# function calculateStats(
#     u_numerical::AbstractVector{<:Real},
#     u_analytical_func::Function,
#     x_coords_input::Union{AbstractVector{<:Real}, AbstractVector{<:NTuple{2,Float64}}},
#     domain_params::NamedTuple,
#     N_particles::Int,
#     stats_to_calculate::Vector{String};
#     dierckx_k::Int = 3, 
#     dierckx_s::Union{Real,Nothing} = nothing,
#     quad_tol::Real = 1e-10, # Increased tolerance slightly for stability
# )::Dict{String, Float64}

#     if N_particles == 0
#         @warn "Numerical solution vector is empty. Returning empty stats."
#         return Dict{String, Float64}()
#     end
#     if length(u_numerical) != N_particles || length(x_coords_input) != N_particles
#         error("Input u_numerical and x_coords_input lengths must match N_particles.")
#     end

#     results = Dict{String, Float64}()
    
#     s_val_actual = isnothing(dierckx_s) ? max(0.0, Float64(N_particles) - sqrt(2.0*Float64(N_particles))) : Float64(dierckx_s)
#     if s_val_actual < 0.0; s_val_actual = 0.0; end

#     # --- 1. Calculate Analytical Solution and Pointwise Errors ---
#     u_analytical_at_particles = Vector{Float64}(undef, N_particles)
#     if eltype(x_coords_input) <: Real # 1D
#         for i in 1:N_particles
#             u_analytical_at_particles[i] = u_analytical_func(x_coords_input[i])
#         end
#     else # 2D (NTuple{2,Float64})
#         for i in 1:N_particles
#             u_analytical_at_particles[i] = u_analytical_func(x_coords_input[i]...)
#         end
#     end
#     errors_at_particles = u_numerical .- u_analytical_at_particles

#     # --- 2. Supremum Norm (Pointwise) ---
#     if "supnorm" in stats_to_calculate || "supnorm" in stats_to_calculate
#         # Sup norm of the error
#         sup_norm_error = N_particles > 0 ? maximum(abs.(errors_at_particles)) : 0.0
#         results["supnorm"] = sup_norm_error # Store absolute sup norm
        
#         # Sup norm of the analytical solution for normalization
#         sup_norm_analytical = N_particles > 0 ? maximum(abs.(u_analytical_at_particles)) : 0.0
#         if sup_norm_analytical > 1e-12
#             results["supnorm"] = sup_norm_error / sup_norm_analytical
#         else
#             results["supnorm"] = sup_norm_error # Avoid division by zero, return absolute error
#         end
#     end

#     # --- 3. Discrete, Unweighted Vector Norms ---
#     if any(s -> occursin("discrete", s), stats_to_calculate)
#         # Discrete L1 error norm: sum of absolute errors
#         discrete_l1_error = sum(abs.(errors_at_particles))
#         results["discrete_l1error"] = discrete_l1_error
        
#         # Discrete L2 error norm: sqrt of sum of squared errors
#         # Using LinearAlgebra.norm is concise and efficient for this.
#         # using LinearAlgebra; discrete_l2_error = norm(errors_at_particles, 2)
#         discrete_l2_error = sqrt(sum(errors_at_particles.^2))
#         results["discrete_l2error"] = discrete_l2_error

#         # Calculate analytical norms for normalization
#         discrete_l1_analytical = sum(abs.(u_analytical_at_particles))
#         discrete_l2_analytical = sqrt(sum(u_analytical_at_particles.^2))

#         if "discrete_l1error" in stats_to_calculate
#             results["discrete_l1error"] = discrete_l1_analytical > 1e-12 ? discrete_l1_error / discrete_l1_analytical : discrete_l1_error
#         end
#         if "discrete_l2error" in stats_to_calculate
#             results["discrete_l2error"] = discrete_l2_analytical > 1e-12 ? discrete_l2_error / discrete_l2_analytical : discrete_l2_error
#         end
#     end

#     # --- 3. Interpolation and Integration for Norms and Mass ---
#     if eltype(x_coords_input) <: Real # 1D Case
#         xmin, xmax = domain_params.xmin, domain_params.xmax
        
#         local x_sorted, u_num_sorted, errors_sorted, u_ana_sorted
#         if N_particles > 1 && !issorted(x_coords_input)
#             perm = sortperm(x_coords_input)
#             x_sorted = x_coords_input[perm]
#             u_num_sorted = u_numerical[perm]
#             errors_sorted = errors_at_particles[perm]
#             u_ana_sorted = u_analytical_at_particles[perm]
#         else
#             x_sorted = x_coords_input
#             u_num_sorted = u_numerical
#             errors_sorted = errors_at_particles
#             u_ana_sorted = u_analytical_at_particles
#         end

#         spl_u_num_lin = Dierckx.Spline1D(x_sorted, u_num_sorted; k=dierckx_k, s=s_val_actual, bc="nearest")
#         spl_u_ana_lin = Dierckx.Spline1D(x_sorted, u_ana_sorted; k=dierckx_k, s=s_val_actual, bc="nearest")
#         spl_u_num = Dierckx.Spline1D(x_sorted, u_num_sorted; k=dierckx_k, s=s_val_actual, bc="nearest")

        
#         # --- Mass Calculation (Spline vs Spline) ---
#         if ("mass" in stats_to_calculate || "relative_mass" in stats_to_calculate) && N_particles > 0
#             mass_num, mass_ana = NaN, NaN
#             mass_num = Dierckx.integrate(spl_u_num_lin, xmin, xmax)
#             #mass_ana = Dierckx.integrate(spl_u_ana_lin, xmin, xmax)
#             mass_ana, _ = QuadGK.quadgk(x -> u_analytical_func(x), xmin, xmax, rtol = quad_tol)
#             if "mass" in stats_to_calculate; results["mass"] = mass_num; end
#             if "relative_mass" in stats_to_calculate
#                 # test_vals = [u_analytical_func(x) for x = xmin:((xmax-xmin)/1000):xmax]
#                 # println("left = ",argmax(test_vals),"right = ",1000 - argmax(reverse(test_vals)),"mass=", mass_ana)
#                 # plot(xmin:((xmax-xmin)/100):xmax,test_vals)
#                 results["relative_mass"] = (abs(mass_ana) > 1e-12 && !isnan(mass_num)) ? mass_num / mass_ana : NaN
#             end
#         end
#         # --- Calculate Absolute Norms of Numerical Solution ---
#         if !isnothing(spl_u_num)
#             if "l1norm" in stats_to_calculate
#                 l1_norm_num, _ = QuadGK.quadgk(x -> abs(spl_u_num(x)), xmin, xmax, rtol=quad_tol)
#                 results["l1norm"] = l1_norm_num
#             end
#             if "l2norm" in stats_to_calculate
#                 l2_sq_norm_num, _ = QuadGK.quadgk(x -> spl_u_num(x)^2, xmin, xmax, rtol=quad_tol)
#                 results["l2norm"] = sqrt(l2_sq_norm_num)
#             end
#         end
#         # --- Integrated Error Norms (Error Spline vs. Analytical Function Integral) ---
#         if ("l1error" in stats_to_calculate || "l2error" in stats_to_calculate) && N_particles > 1
#             spl_error_1D = nothing
#             try spl_error_1D = Dierckx.Spline1D(x_sorted, errors_sorted; k=dierckx_k, s=s_val_actual, bc="nearest")
#             catch e; @warn "Dierckx.Spline1D for error failed: $e. Integrated L1/L2 norms will be NaN."; end

#             if !isnothing(spl_error_1D)
#                 # Normalization denominator is the norm of the TRUE analytical function
#                 ana_l1_norm_integrated, _ = QuadGK.quadgk(x -> abs(u_analytical_func(x)), xmin, xmax, rtol=quad_tol)
#                 ana_l2_norm_sq_integrated, _ = QuadGK.quadgk(x -> u_analytical_func(x)^2, xmin, xmax, rtol=quad_tol)
#                 ana_l2_norm_integrated = sqrt(ana_l2_norm_sq_integrated)

#                 if "l1error" in stats_to_calculate
#                     l1_err_val, _ = QuadGK.quadgk(x -> abs(spl_error_1D(x)), xmin, xmax, rtol=quad_tol)
#                     results["l1error"] = ana_l1_norm_integrated > 1e-12 ? l1_err_val / ana_l1_norm_integrated : l1_err_val
#                 end
#                 if "l2error" in stats_to_calculate
#                     l2_err_sq_val, _ = QuadGK.quadgk(x -> spl_error_1D(x)^2, xmin, xmax, rtol=quad_tol)
#                     results["l2error"] = ana_l2_norm_integrated > 1e-12 ? sqrt(l2_err_sq_val) / ana_l2_norm_integrated : sqrt(l2_err_sq_val)
#                 end
#             else # spl_error_1D failed to be created
#                  if "l1error" in stats_to_calculate; results["l1error"] = NaN; end
#                  if "l2error" in stats_to_calculate; results["l2error"] = NaN; end
#             end
#         end
#     elseif eltype(x_coords_input) <: NTuple{2,Float64} # 2D Case
#         @warn "Integrated L1/L2 norms for 2D are not yet implemented with a high-order quadrature package. Discrete norms can be used as a proxy."
#         # If stats_to_calculate contains "l1norm" or "l2norm", report NaN as they are not computed.
#         if "l1norm" in stats_to_calculate; results["l1norm"] = NaN; end
#         if "l2norm" in stats_to_calculate; results["l2norm"] = NaN; end
#         # Mass calculation for 2D via Dierckx can still be done
#         if "mass" in stats_to_calculate && N_particles > 0
#             xmin, xmax = domain_params.xmin, domain_params.xmax
#             ymin, ymax = domain_params.ymin, domain_params.ymax
#             x_vec_2d = [pt[1] for pt in x_coords_input]
#             y_vec_2d = [pt[2] for pt in x_coords_input]
#             try
#                 spl_u_num_2D = Dierckx.Spline2D(x_vec_2d, y_vec_2d, u_numerical; kx=dierckx_k, ky=dierckx_k, s=s_val_actual)
#                 results["mass"] = Dierckx.integrate(spl_u_num_2D, xmin, xmax, ymin, ymax)
#             catch e
#                 @warn "Dierckx.Spline2D for u_numerical (mass) failed: $e. Mass not computed."
#                 results["mass"] = NaN
#             end
#         elseif "mass" in stats_to_calculate
#             results["mass"] = 0.0
#         end
#     end

#     # --- Wave Tracking (Pointwise) ---
#     if ("wave_position_error" in stats_to_calculate || "wave_height_error" in stats_to_calculate) && N_particles > 0
#         height_ana, index_ana = findmax(u_analytical_at_particles)
#         pos_ana = x_coords_input[index_ana]
#         height_num, index_num = findmax(u_numerical)
#         pos_num = x_coords_input[index_num]
        
#         if "wave_position_error" in stats_to_calculate
#             domain_length = eltype(x_coords_input) <: Real ? (domain_params.xmax - domain_params.xmin) : 1.0 # Placeholder for 2D domain size
#             results["wave_position_error"] = domain_length > 1e-9 ? abs(pos_ana - pos_num) / domain_length : abs(pos_ana - pos_num)
#         end
#         if "wave_height_error" in stats_to_calculate
#             results["wave_height_error"] = abs(height_ana) > 1e-9 ? abs(height_ana - height_num) / abs(height_ana) : abs(height_ana - height_num)
#         end
#     end
    
#     return results
# end

# function calculateAllStats!(
#         sim_data::SimData1D,
#         u_ana_func::Function,
#         domain_params::NamedTuple,
#         N::Int;
#         dierckx_k::Int = 3, 
#         dierckx_s::Union{Real,Nothing} = nothing,
#         quad_tol::Real = 1e-9,
#         stats_to_calculate::Union{String,Vector{String}} = "all"
#     )
#     if stats_to_calculate == "all"
#         stats_to_calculate = [
#         "l1norm", "l2norm", "supnorm", "mass", "relative_mass",
#         "wave_position_error", "wave_height_error",
#         "discrete_l2error", "discrete_l1error", "l1error", "l2error"
#     ]
#     end
#     for key = stats_to_calculate
#         sim_data.stats[key] = []
#     end
#     for (m,t) = enumerate(sim_data.t)
#         if m == length(sim_data.t)
#             t = sim_data.params["tmax"]
#         end
#         stats_tmp = calculateStats(sim_data.u[m], 
#                                    x -> u_ana_func(x, t), 
#                                    sim_data.x[m], 
#                                    domain_params, N, stats_to_calculate; 
#                                    quad_tol = quad_tol, 
#                                    dierckx_k = dierckx_k, 
#                                    dierckx_s = dierckx_s)
#         for (key,val) = stats_tmp
#             push!(sim_data.stats[key],val)
#         end
#     end

# end
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