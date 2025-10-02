# ============== Suggested new file: SourceTerms.jl ==============
module SourceTerms

using ..HyperbolicPDEs

export AbstractSourceTerm, RelaxationSourceTerm1D, RelaxationSourceTerm, MaxwellianFunctor

abstract type AbstractSourceTerm end


# Helper for 1D PDEs: flux_result is a single tuple
@inline function get_flux_component(flux_result::Tuple, i_macro::Int, i_dim::Int, ::Val{1})
    return flux_result[i_macro]
end

# Helper for 2D PDEs: flux_result is a tuple of tuples
@inline function get_flux_component(flux_result::Tuple, i_macro::Int, i_dim::Int, ::Val{2})
    return flux_result[i_dim][i_macro]
end

# Helper for 1D PDEs: flux_result is a single tuple
@inline function get_flux_component(flux_result::Float64, i_dim::Int, ::Val{1})
    return flux_result
end

# Helper for 2D PDEs: flux_result is a tuple of tuples
@inline function get_flux_component(flux_result::Tuple, i_dim::Int, ::Val{2})
    return flux_result[i_dim]
end

# F is a parameter for the concrete type of the flux function.
struct MaxwellianFunctor{D,N,E <: HyperbolicPDE{D, N}}
    system_eq::E
    i_macro::Int64
    i_dim::Int64
    relax_speed::Float64
    coefficient::Float64
    interior_factor::Float64
end

# --- Method 1: Specialized for SCALAR PDEs (N=1) ---

# This method is only called if the functor's type is MaxwellianFunctor{D, 1, E}
function (m::MaxwellianFunctor{D, 1, E})(U)::Float64 where {D, E}
    # For a scalar equation, U is a 1-tuple, e.g., (rho,). We extract the value.
    u_scalar = U[1]
    
    # The flux function for a scalar PDE expects a single Float64
    flux_result = flux(m.system_eq, u_scalar)
    
    # Use the helper to handle 1D (scalar) vs 2D (tuple) flux results
    flux_val = get_flux_component(flux_result, m.i_dim, Val(D))

    # Note: For scalar relaxation, U[m.i_macro] is just u_scalar.
    return m.coefficient * (u_scalar + m.interior_factor * flux_val / m.relax_speed)
end


# --- Method 2: General for SYSTEM PDEs (any N, any D) ---

# This method is called for any MaxwellianFunctor, but because we defined a more
# specific one for N=1, this one will be used for all N > 1 cases.
function (m::MaxwellianFunctor{D, N, E})(U)::Float64 where {D, N, E}
    macro_val = U[m.i_macro]
    
    # The flux function for a system PDE expects a tuple
    flux_result = flux(m.system_eq, U)
    
    # Use the helper to handle 1D vs 2D system flux results
    flux_val = get_flux_component(flux_result, m.i_macro, m.i_dim, Val(D))

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
    particle_pos::Any,
    time::Any
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
    # The buffer now has a concrete type!
    macro_buffer::Vector{Float64}
end

function RelaxationSourceTerm(
    maxwellian_functions_input::AbstractVector{<:MaxwellianFunctor},
    epsilon::Float64,
    kinetic_indices_input::AbstractVector{<:AbstractVector{Int}}
)   
    maxwellian_tuple = Tuple(maxwellian_functions_input)
    kinetic_indices_tuple = Tuple(Tuple(indices) for indices in kinetic_indices_input)
    num_total_kin = length(maxwellian_tuple)
    num_macro_vars = length(kinetic_indices_tuple)
    
    return RelaxationSourceTerm(
        maxwellian_tuple, 
        kinetic_indices_tuple,
        epsilon,
        num_total_kin,
        num_macro_vars,
        # Initialize the buffer with a concrete type
        Vector{Float64}(undef, num_macro_vars)
    )
end

function (rs::RelaxationSourceTerm{MF,KI})(
    S_out_particle::AbstractVector{Float64},
    U_kinetic_particle::AbstractVector{Float64},
    particle_pos::Tuple{Float64,Float64}, 
    time::Float64             
) where {MF <: Tuple, KI <: Tuple}
    if length(U_kinetic_particle) != rs.num_total_kinetic_components || length(S_out_particle) != rs.num_total_kinetic_components
        error("Dimension mismatch in RelaxationSourceTerm functor.")
    end

    for i = 1:rs.num_macro_variables
        rs.macro_buffer[i] = sum(U_kinetic_particle[k] for k in rs.kinetic_indices[i])
    end
    for (k_global_comp, maxwellian) in enumerate(rs.maxwellians)

        mk_of_U_macro = maxwellian(rs.macro_buffer)
        
        S_out_particle[k_global_comp] = (mk_of_U_macro - U_kinetic_particle[k_global_comp]) / rs.epsilon
    end
end


end # Module SourceTerms