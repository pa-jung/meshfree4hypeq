# ============== Suggested new file: SourceTerms.jl ==============
module SourceTerms

# If Maxwellians need physical flux F(u) and it's defined in a shared module:
# Example: using ..Meshfree4ScalarEq.ScalarHyperbolicEquations # Adjust path

export AbstractSourceTerm, RelaxationSourceTerm1D, RelaxationSourceTerm

abstract type AbstractSourceTerm end

"""
    (source::AbstractSourceTerm)(
        S_out_particle::AbstractVector{Float64}, # Output vector S(U)
        U_particle::AbstractVector{Float64},     # Input state vector U
        particle_pos::Float64,
        time::Real
    )

Functor interface for source terms. Modifies `S_out_particle` in place.
"""
function (source::AbstractSourceTerm)(
    S_out_particle::AbstractVector{Float64},
    U_particle::AbstractVector{Float64},
    particle_pos::Float64,
    time::Real
)
    error("Functor () not implemented for source term type $(typeof(source))")
end

struct RelaxationSourceTerm1D{MF <: AbstractVector{<:Function}} <: AbstractSourceTerm
    maxwellians::MF
    epsilon::Float64
    num_components::Int

    function RelaxationSourceTerm1D(
        maxwellian_functions::MF,
        epsilon::Float64
    ) where {MF <: AbstractVector{<:Function}}
        if isempty(maxwellian_functions)
            error("Maxwellian functions vector cannot be empty.")
        end
        if epsilon <= 0.0
            error("Relaxation parameter epsilon must be positive.")
        end
        new{MF}(maxwellian_functions, epsilon, length(maxwellian_functions))
    end
end

function (rs::RelaxationSourceTerm1D)(
    S_out_particle::AbstractVector{Float64},
    U_particle::AbstractVector{Float64},
    particle_pos::Float64, # Unused by this specific S
    time::Real             # Unused by this specific S
)
    if length(U_particle) != rs.num_components || length(S_out_particle) != rs.num_components
        error("Dimension mismatch in RelaxationSourceTerm. Expected $(rs.num_components) components. Got U: $(length(U_particle)), S_out: $(length(S_out_particle))")
    end

    rho_total = 0.0
    for k_comp in 1:rs.num_components
        rho_total += U_particle[k_comp]
    end

    for k_comp in 1:rs.num_components
        mk_of_rho = rs.maxwellians[k_comp](rho_total)
        S_out_particle[k_comp] = (mk_of_rho - U_particle[k_comp]) / rs.epsilon
    end
end

struct RelaxationSourceTerm{MF <: AbstractVector{<:Function}, VI <: AbstractVector{<:AbstractVector{Int}}} <: AbstractSourceTerm
    maxwellians::MF         # M_k(u_macro_1, u_macro_2, ...)
    epsilon::Float64
    kinetic_indices::VI # Defines how kinetic components sum to form each u_macro_j
    num_total_kinetic_components::Int  # Total number of v_k
    num_macro_variables::Int           # Number of arguments M_k expects (length of U_macro vector)

    # Constructor for general coupled systems
    function RelaxationSourceTerm(
        maxwellian_functions::MF,
        epsilon::Float64,
        kinetic_indices::VI
    ) where {MF <: AbstractVector{<:Function}, VI <: AbstractVector{<:AbstractVector{Int}}}
        # ... (parameter checks as before) ...
        num_total_kin = length(maxwellian_functions)
        num_macro_vars = length(kinetic_indices)
        # For coupled systems, num_macro_vars <= num_total_kin generally.
        # Each M_k will receive `num_macro_vars` arguments.
        new{MF, VI}(maxwellian_functions, epsilon, kinetic_indices, num_total_kin, num_macro_vars)
    end

    # Constructor for "scalar-like" relaxation: each v_k relaxes towards M_k(v_k)
    function RelaxationSourceTerm(
        maxwellian_functions_scalar_like::MF, # Each M_k here should expect 1 argument (its own v_k)
        epsilon::Float64
    ) where {MF <: AbstractVector{<:Function}}
        # ... (parameter checks) ...
        num_total_kin = length(maxwellian_functions_scalar_like)
        # For this case, each kinetic var is its own "macro" variable for its Maxwellian
        kinetic_map = [1:num_total_kin] 
        num_macro_vars_effective = 1 # Each M_k effectively sees a "macro state" of length 1 (just itself)
        
        println("INFO: RelaxationSourceTerm created for scalar-like relaxation. Each M_k will be called with its corresponding v_k as the single argument.")
        # We use the main constructor, but the flag indicates special handling in functor
        new{MF, typeof(kinetic_map)}(maxwellian_functions_scalar_like, epsilon, kinetic_map, num_total_kin, num_macro_vars_effective)
    end
end

function (rs::RelaxationSourceTerm)(
    S_out_particle::AbstractVector{Float64},
    U_kinetic_particle::AbstractVector{Float64},
    particle_pos::Float64, 
    time::Real             
)
    if length(U_kinetic_particle) != rs.num_total_kinetic_components || length(S_out_particle) != rs.num_total_kinetic_components
        error("Dimension mismatch in RelaxationSourceTerm functor.")
    end

    # General coupled case: reconstruct full U_macro vector
    U_macro_particle_values = Vector{Float64}(undef, rs.num_macro_variables)
    for i_macro in 1:rs.num_macro_variables
        U_macro_particle_values[i_macro] = sum(U_kinetic_particle[rs.kinetic_indices[i_macro]])
    end
    
    for k_global_comp in 1:rs.num_total_kinetic_components
        # Maxwellian rs.maxwellians[k_global_comp] expects rs.num_macro_variables arguments
        mk_of_U_macro = rs.maxwellians[k_global_comp](U_macro_particle_values...) # Splat
        S_out_particle[k_global_comp] = (mk_of_U_macro - U_kinetic_particle[k_global_comp]) / rs.epsilon
    end
end


end # Module SourceTerms