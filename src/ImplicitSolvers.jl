# ============== Suggested new file: ImplicitSolvers.jl ==============
module ImplicitSolvers

using ..SourceTerms

# To use AbstractSourceTerm here, ensure it's accessible:
# using ..SourceTerms # If SourceTerms.jl is at the same level or correctly pathed

# For this example, we'll use `Any` for source_term_object and rely on its functor interface
# A more type-safe approach would be: source_term_object::ST where ST <: AbstractSourceTerm

export AbstractImplicitSolver, PicardIterationSolver, LinearizedRelaxationImplicitSolver, solve!

abstract type AbstractImplicitSolver end

struct PicardIterationSolver <: AbstractImplicitSolver
    max_iters::Int
    tol::Float64

    function PicardIterationSolver(;max_iters::Int = 20, tol::Float64 = 1e-8)
        new(max_iters, tol)
    end
end

function solve!(
    solver::PicardIterationSolver,
    Y_out_particle::AbstractVector{Float64}, 
    RHS_const_particle::AbstractVector{Float64},
    dt_coefficient_for_S::Float64,
    source_term_object, # Can be AbstractSourceTerm or just a callable
    particle_pos::Float64,
    time_for_S_eval::Real,
    N_components::Int
)::Bool
    if N_components == 0 && length(Y_out_particle) == 0
        return true 
    end
    if length(Y_out_particle) != N_components || length(RHS_const_particle) != N_components
        error("Vector size mismatch in PicardIterationSolver.solve!")
    end

    S_eval_local = Vector{Float64}(undef, N_components)
    Y_prev_iter = similar(Y_out_particle)
    converged = false
    norm_diff::Float64 = Inf 
    
    for iter in 1:solver.max_iters
        Y_prev_iter .= Y_out_particle
        source_term_object(S_eval_local, Y_out_particle, particle_pos, time_for_S_eval)
        
        for k_comp in 1:N_components
            Y_out_particle[k_comp] = RHS_const_particle[k_comp] + dt_coefficient_for_S * S_eval_local[k_comp]
        end
        
        norm_diff = 0.0
        for k_comp in 1:N_components
            norm_diff = max(norm_diff, abs(Y_out_particle[k_comp] - Y_prev_iter[k_comp]))
        end

        if norm_diff < solver.tol
            converged = true
            break
        end
    end

    if !converged
        @warn "PicardIterationSolver did not converge for particle at pos $particle_pos, time $time_for_S_eval after $(solver.max_iters) iterations. Max Diff: $norm_diff"
    end
    return converged
end

# --- NEW: LinearizedRelaxationImplicitSolver ---
"""
    LinearizedRelaxationImplicitSolver <: AbstractImplicitSolver

Solves the implicit part for a relaxation source term of the form 
S_k = (M_k(rho_base) - U_k_base)/epsilon using a linearized, direct update:
U_k_new = (U_k_base + (dt'/epsilon)*M_k(rho_base)) / (1 + dt'/epsilon),
where rho_base = sum(U_k_base) from the state before this implicit step.
This is a non-iterative update for U_k once M_k(rho_base) is computed.
"""
struct LinearizedRelaxationImplicitSolver <: AbstractImplicitSolver
    # No internal fields needed if all info comes from source_term_object and arguments
    function LinearizedRelaxationImplicitSolver()
        new()
    end
end

function solve!(
    solver::LinearizedRelaxationImplicitSolver,
    Y_out_particle::AbstractVector{Float64},         # Output: result U_k_new is stored here
    RHS_const_particle::AbstractVector{Float64},     # Input: This is U_k_base (e.g., U^n or U_temp for ARS2 stages)
    dt_coefficient_for_S::Float64,              # This is the effective dt' (e.g., dt*gamma in ARS2)
    source_term_object::RelaxationSourceTerm1D,     # Must be RelaxationSourceTerm
    particle_pos::Float64,                      # Unused by this specific solver for this source
    time_for_S_eval::Real,                      # Unused if Maxwellians are not time-dependent
    N_components::Int
)::Bool                                          # Always "converges" in one step for this direct formula

    if N_components == 0 && length(Y_out_particle) == 0
        return true # Nothing to solve
    end
    if length(Y_out_particle) != N_components || length(RHS_const_particle) != N_components
        error("Vector size mismatch in LinearizedRelaxationImplicitSolver.solve!")
    end

    epsilon = source_term_object.epsilon
    maxwellians = source_term_object.maxwellians # Vector of M_k functions

    # 1. Calculate rho_base = sum(U_k_base) using RHS_const_particle
    #    (which is the state *before* this current implicit relaxation step)
    rho_base = 0.0
    for k_comp in 1:N_components
        rho_base += RHS_const_particle[k_comp]
    end

    # 2. Apply the direct update formula for each component
    #    Y_k_new = (epsilon * U_k_base + dt' * M_k(rho_base)) / (epsilon + dt')
    #    where U_k_base is RHS_const_particle[k_comp]
    #    and dt' is dt_coefficient_for_S

    coeff_sum_inv = 1.0 / (epsilon + dt_coefficient_for_S) # Precompute for efficiency

    for k_comp in 1:N_components
        U_k_base = RHS_const_particle[k_comp]
        Mk_of_rho_base = maxwellians[k_comp](rho_base)
        
        Y_out_particle[k_comp] = (epsilon * U_k_base + dt_coefficient_for_S * Mk_of_rho_base) * coeff_sum_inv
    end
    
    return true # Direct formula, always "converges" in one evaluation
end

function solve!(
    solver::LinearizedRelaxationImplicitSolver,
    Y_out_particle::AbstractVector{Float64},         
    RHS_const_particle::AbstractVector{Float64},     
    dt_coefficient_for_S::Float64,              
    source_term_object::RelaxationSourceTerm,     
    particle_pos::Float64,                      
    time_for_S_eval::Real,                      
    N_total_kinetic_components_arg::Int      
)::Bool
    # ... (checks) ...
    epsilon = source_term_object.epsilon
    maxwellians = source_term_object.maxwellians

    coeff_sum_inv = 1.0 / (epsilon + dt_coefficient_for_S)

    # General coupled case: reconstruct U_macro_base
    kinetic_map = source_term_object.kinetic_indices
    N_macro_vars = source_term_object.num_macro_variables
    U_macro_base_values = Vector{Float64}(undef, N_macro_vars)
    for i_macro in 1:N_macro_vars
        U_macro_base_values[i_macro] = sum(RHS_const_particle[kinetic_map[i_macro]])
    end

    for k_global_comp in 1:N_total_kinetic_components_arg
        v_k_base_kinetic = RHS_const_particle[k_global_comp]
        # M_k here expects N_macro_vars arguments, splatted from U_macro_base_values
        Mk_val = maxwellians[k_global_comp](U_macro_base_values...)
        Y_out_particle[k_global_comp] = (epsilon * v_k_base_kinetic + dt_coefficient_for_S * Mk_val) * coeff_sum_inv
    end
    return true 
end

end # Module ImplicitSolvers