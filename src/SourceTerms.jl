# ============== Suggested new file: SourceTerms.jl ==============
module SourceTerms

using ..HyperbolicPDEs

export AbstractSourceTerm, RelaxationSourceTerm1D, RelaxationSourceTerm, MaxwellianFunctor

abstract type AbstractSourceTerm end

# F: Type of the captured flux function.
# ============== CORRECTED VERSION ==============

# F is a parameter for the concrete type of the flux function.
struct MaxwellianFunctor{E <: HyperbolicPDE}
    system_eq::E
    i_macro::Int64
    i_dim::Int64
    relax_speed::Float64
    coefficient::Float64
    interior_factor::Float64
end

# The signature now includes the {F} parameter.
function (m::MaxwellianFunctor{E})(U::Tuple{Vararg{Float64}})::Float64 where {E <: HyperbolicPDE}
    macro_val = U[m.i_macro]
    flux_val = flux(m.system_eq, U)[m.i_dim][m.i_macro]

    # Use the struct field `m.interior_factor` in the calculation
    return m.coefficient * (macro_val + m.interior_factor * flux_val / m.relax_speed)
end



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
    particle_pos::Any, # Unused by this specific S
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

struct RelaxationSourceTerm{MF <: Tuple, KI <: Tuple} <: AbstractSourceTerm
    maxwellians::MF
    kinetic_indices::KI
    epsilon::Float64
    num_total_kinetic_components::Int64
    num_macro_variables::Int64
end
# This is the user-friendly constructor that accepts Vectors
function RelaxationSourceTerm(
    maxwellian_functions_input::AbstractVector{<:MaxwellianFunctor},
    epsilon::Float64,
    kinetic_indices_input::AbstractVector{<:AbstractVector{Int}}
)   
    # Convert the input vectors to tuples
    maxwellian_tuple = Tuple(maxwellian_functions_input)

    # A cleaner way to convert the nested vector to a tuple of tuples
    kinetic_indices_tuple = Tuple(Tuple(indices) for indices in kinetic_indices_input)

    # Calculate the remaining properties
    num_total_kin = length(maxwellian_tuple)
    num_macro_vars = length(kinetic_indices_tuple)
    
    # Call the default constructor with the performant tuple types.
    # Julia will automatically create a concrete instance, e.g.:
    # RelaxationSourceTerm{Tuple{MaxwellianFunctor{...}}, NTuple{...}}(...)
    return RelaxationSourceTerm(
        maxwellian_tuple, 
        kinetic_indices_tuple,
        epsilon,
        num_total_kin,
        num_macro_vars
    )
end

function (rs::RelaxationSourceTerm{MF,KI})(
    S_out_particle::AbstractVector{Float64},
    U_kinetic_particle::AbstractVector{Float64},
    particle_pos::Any, 
    time::Real             
) where {MF <: Tuple, KI <: Tuple}
    if length(U_kinetic_particle) != rs.num_total_kinetic_components || length(S_out_particle) != rs.num_total_kinetic_components
        error("Dimension mismatch in RelaxationSourceTerm functor.")
    end

    U_macro_tuple = ntuple(rs.num_macro_variables) do i_macro
        # The `sum` needs a generator `(x for x in ...)` to be fast inside `ntuple`
        sum(U_kinetic_particle[k] for k in rs.kinetic_indices[i_macro])
    end
    
    for k_global_comp in 1:rs.num_total_kinetic_components
        # Pass the TUPLE to the Maxwellian
        mk_of_U_macro = rs.maxwellians[k_global_comp](U_macro_tuple) 
        S_out_particle[k_global_comp] = (mk_of_U_macro - U_kinetic_particle[k_global_comp]) / rs.epsilon
    end
end


end # Module SourceTerms