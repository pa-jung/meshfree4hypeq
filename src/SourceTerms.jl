# ============== Suggested new file: SourceTerms.jl ==============
module SourceTerms

# If Maxwellians need physical flux F(u) and it's defined in a shared module:
# Example: using ..Meshfree4ScalarEq.ScalarHyperbolicEquations # Adjust path

export AbstractSourceTerm, RelaxationSourceTerm

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

struct RelaxationSourceTerm{MF <: AbstractVector{<:Function}} <: AbstractSourceTerm
    maxwellians::MF
    epsilon::Float64
    num_components::Int

    function RelaxationSourceTerm(
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

function (rs::RelaxationSourceTerm)(
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

end # Module SourceTerms