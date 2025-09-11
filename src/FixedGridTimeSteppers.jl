export Upwind, LaxFriedrich, ClassicalTimeStepper, ClassicalRK2LWTimeStepper, ClassicalRichtmyerLWMOOD
using ..Meshfree4ScalarEq.FluxFunctions

struct Upwind <: FixedGridTimeStepper 
    rhoOld::Vector{Float64}
    function Upwind(Nx::Integer)
        new(Vector{Float64}(undef, Nx))
    end
end

function (upwind::Upwind)(eq::HyperbolicPDE, particleGrid::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    error("Upwind method for nonlinear hyperbolic equations (Roe's scheme) not yet implemented.")
end

function (upwind::Upwind)(eq::LinearAdvection{1}, particleGrid::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    map!(particle -> particle.rho, upwind.rhoOld, particleGrid.grid)
    vel = velocity(eq, particleGrid.grid[1].rho)  # Velocity is constant so just evaluate it at the first particle
    λ = vel*dt/particleGrid.dx
    if vel > 0
        particleGrid.grid[1].rho -= λ*(upwind.rhoOld[1] - upwind.rhoOld[end])
        for i in 2:particleGrid.N
            particleGrid.grid[i].rho -= λ*(upwind.rhoOld[i] - upwind.rhoOld[i-1])
        end
    else
        for i in 1:particleGrid.N-1
            particleGrid.grid[i].rho -= λ*(upwind.rhoOld[i+1] - upwind.rhoOld[i])
        end
        particleGrid.grid[end].rho -= λ*(upwind.rhoOld[1] - upwind.rhoOld[end])
    end
end

struct LaxFriedrich <: FixedGridTimeStepper 
    rhoOld::Vector{Float64}
    function LaxFriedrich(Nx::Integer)
        new(Vector{Float64}(undef, Nx))
    end
end

function (lf::LaxFriedrich)(eq::ScalarHyperbolicPDE{1}, particleGrid::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    map!(particle -> particle.rho, lf.rhoOld, particleGrid.grid)
    λ = dt/(2*particleGrid.dx)
    for i in 2:particleGrid.N-1
        particleGrid.grid[i].rho = 0.5*(lf.rhoOld[i+1] + lf.rhoOld[i-1]) -λ*(flux(eq, lf.rhoOld[i+1]) - flux(eq, lf.rhoOld[i-1]))
    end
    particleGrid.grid[end].rho = 0.5*(lf.rhoOld[1] + lf.rhoOld[end-1]) -λ*(flux(eq, lf.rhoOld[1]) - flux(eq, lf.rhoOld[end-1]))
    particleGrid.grid[1].rho = 0.5*(lf.rhoOld[2] + lf.rhoOld[end]) -λ*(flux(eq, lf.rhoOld[2]) - flux(eq, lf.rhoOld[end]))
end

# --- NEW: ClassicalTimeStepper (1st Order Finite Volume) ---
# This version directly uses a NumericalFluxFunction from your FluxFunctions.txt

struct ClassicalTimeStepper <: FixedGridTimeStepper
    numericalFlux::NumericalFluxFunction # Instance of RusanovFlux(), UpwindFlux(), etc.
    rhoOld::Vector{Float64}
    # Buffers for interface fluxes can be local if preferred, or fields if used elsewhere
    # For simplicity here, let's make them local to the functor call.

    function ClassicalTimeStepper(Nx::Integer, numFlux::NumericalFluxFunction)
        new(numFlux, Vector{Float64}(undef, Nx))
    end
end

# In your TimeIntegration.jl or FixedGridTimeSteppers.txt file

# (Keep the ClassicalTimeStepper struct definition as is)
# struct ClassicalTimeStepper <: FixedGridTimeStepper ... end

"""
    (cts::ClassicalTimeStepper)(eq, particleGrid, settings, time, dt)

Functor for the Classical Finite Volume (Fixed Grid) Time Stepper.
Handles both periodic and fixed boundary conditions.
"""
function (cts::ClassicalTimeStepper)(
    eq::ScalarHyperbolicPDE{1}, 
    particleGrid::ParticleGrid1D, 
    settings::SimSetting, 
    time::Real, 
    dt::Real
)
    # This timestepper is designed for regular grids.
    if !particleGrid.regular
        @warn "ClassicalTimeStepper is designed for regular grids but was called with a non-regular one. Results may be inaccurate."
    end

    # Copy the initial state for all particles (including ghosts) into the buffer
    map!(particle -> particle.rho, cts.rhoOld, particleGrid.grid)
    
    dx = particleGrid.dx
    dtdx = dt / dx

    if particleGrid.bc == :periodic
        # --- Periodic Boundary Condition Logic (Original Code) ---
        N = particleGrid.N # Number of physical particles
        
        # flux_at_interfaces[k] will store F*_{k+1/2}
        flux_at_interfaces = Vector{Float64}(undef, N)

        for k in 1:N
            u_L = cts.rhoOld[k]
            u_R = cts.rhoOld[mod1(k + 1, N)] # Periodic neighbor
            flux_at_interfaces[k] = cts.numericalFlux(u_L, u_R, eq)
        end

        for i in 1:N
            F_star_i_plus_half = flux_at_interfaces[i]
            F_star_i_minus_half = flux_at_interfaces[mod1(i - 1, N)] # Periodic neighbor
            particleGrid.grid[i].rho = cts.rhoOld[i] - dtdx * (F_star_i_plus_half - F_star_i_minus_half)
        end

    else # --- Fixed Boundary Condition Logic (e.g., :fixed_dirichlet, :outflow) ---
        N_total = length(particleGrid.grid)
        interior_indices = particleGrid.interior_indices
        
        # We need to calculate fluxes at N_interior + 1 interfaces.
        # These are the interfaces bounding the interior cells.
        # Let's calculate all N_total - 1 interface fluxes for simplicity.
        flux_at_interfaces = Vector{Float64}(undef, N_total - 1)

        # Calculate all interface fluxes F*_{i+1/2} for i = 1 to N_total-1
        for i in 1:(N_total - 1)
            u_L = cts.rhoOld[i]
            u_R = cts.rhoOld[i + 1]
            flux_at_interfaces[i] = cts.numericalFlux(u_L, u_R, eq)
        end

        # Update rule: U_i^{n+1} = U_i^n - (dt/dx) * ( F*_{i+1/2} - F*_{i-1/2} )
        # Loop ONLY over the interior physical particles
        for i in interior_indices
            # F*_{i+1/2} is the flux at the right interface of cell i.
            # In our 0-based thinking, this is interface `i`.
            F_star_i_plus_half = flux_at_interfaces[i]
            
            # F*_{i-1/2} is the flux at the left interface of cell i.
            # This is interface `i-1`.
            F_star_i_minus_half = flux_at_interfaces[i - 1]
            
            particleGrid.grid[i].rho = cts.rhoOld[i] - dtdx * (F_star_i_plus_half - F_star_i_minus_half)
        end
    end
end

# --- NEW: ClassicalRK2LWTimeStepper (Richtmyer two-step Lax-Wendroff) ---
# This one remains the same as it uses the physical flux F(U) on predicted states,
# not a generic NumericalFluxFunction for its core logic.
struct ClassicalRK2LWTimeStepper <: FixedGridTimeStepper
    rhoOld::Vector{Float64}
    rhoPredict_interface::Vector{Float64} # U_{i+1/2}^{n+1/2} - N values for N interfaces
    # No need for fluxPredict as a field, can be local

    function ClassicalRK2LWTimeStepper(Nx::Integer)
        new(Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx))
    end
end

function (crk2::ClassicalRK2LWTimeStepper)(eq::ScalarHyperbolicPDE{1}, particleGrid::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    map!(particle -> particle.rho, crk2.rhoOld, particleGrid.grid)
    N = particleGrid.N
    dx = particleGrid.dx
    
    # --- Predictor Step: Calculate U_{i+1/2}^{n+1/2} ---
    # This is stored in crk2.rhoPredict_interface[i] (representing interface i+1/2)
    for i in 1:N
        idx_plus_1 = mod1(i + 1, N) # Periodic neighbor for U_{i+1}^n
        
        Ui_n = crk2.rhoOld[i]
        Uip1_n = crk2.rhoOld[idx_plus_1]
        
        F_Ui_n = flux(eq, Ui_n)         # Physical flux F(U_i^n)
        F_Uip1_n = flux(eq, Uip1_n)     # Physical flux F(U_{i+1}^n)
        
        # U_{i+1/2}^{n+1/2}
        crk2.rhoPredict_interface[i] = 0.5 * (Ui_n + Uip1_n) - (dt / (2.0 * dx)) * (F_Uip1_n - F_Ui_n)
    end

    # --- Corrector Step ---
    # Temporary storage for fluxes F(U_{i+1/2}^{n+1/2})
    flux_of_predicted_interface_states = Vector{Float64}(undef, N)
    for i in 1:N
        # flux_of_predicted_interface_states[i] is F^*_{i+1/2}
        flux_of_predicted_interface_states[i] = flux(eq, crk2.rhoPredict_interface[i])
    end

    # Update U_i^{n+1}
    for i in 1:N
        idx_minus_1 = mod1(i - 1, N) # For F^*_{i-1/2}
        
        F_star_ip_half = flux_of_predicted_interface_states[i]             # This is F(U_{i+1/2}^{n+1/2})
        F_star_im_half = flux_of_predicted_interface_states[idx_minus_1]   # This is F(U_{(i-1)+1/2}^{n+1/2}) = F(U_{i-1/2}^{n+1/2})
        
        particleGrid.grid[i].rho = crk2.rhoOld[i] - (dt / dx) * (F_star_ip_half - F_star_im_half)
    end
end


# --- Helper function for Lax-Friedrichs dissipation ---
# This can be a local helper function if only used here.
"""
    _max_abs_speed_classical(eq, uL, uR)

Calculates the maximum absolute wavespeed for the state between uL and uR.
Used as the dissipation coefficient for the Lax-Friedrichs flux.
"""
function _max_abs_speed_classical(eq::LinearAdvection, uL::Real, uR::Real)
    return abs(eq.vel)
end

function _max_abs_speed_classical(eq::BurgersEquation, uL::Real, uR::Real)
    # For Burger's equation, the characteristic speed is u.
    return max(abs(uL), abs(uR))
end
# Add methods for other equations like Euler if needed.


# --- NEW: Classical Richtmyer Lax-Wendroff with MOOD ---

struct ClassicalRichtmyerLWMOOD{M <: MOODCriterion} <: FixedGridTimeStepper
    mood::M
    rhoOld::Vector{Float64}
    rhoCandidate::Vector{Float64}         # Buffer for the high-order candidate solution
    rhoPredict_interface::Vector{Float64} # Buffer for U_{i+1/2}^{n+1/2}

    function ClassicalRichtmyerLWMOOD(Nx_total::Integer; mood::M = NoMOOD()) where {M <: MOODCriterion}
        # Buffers need to be sized for the total grid size, including ghosts
        new{M}(mood, Vector{Float64}(undef, Nx_total), Vector{Float64}(undef, Nx_total), Vector{Float64}(undef, Nx_total))
    end
end

function initTimeStepper(cts_mood::ClassicalRichtmyerLWMOOD, particleGrid::ParticleGrid, settings::SimSetting)
    # Resize buffers if grid size changes between runs
    N_total = length(particleGrid.grid)
    if length(cts_mood.rhoOld) != N_total
        resize!(cts_mood.rhoOld, N_total)
        resize!(cts_mood.rhoCandidate, N_total)
        resize!(cts_mood.rhoPredict_interface, N_total)
    end
    return
end

function (cts_mood::ClassicalRichtmyerLWMOOD)(
    eq::ScalarHyperbolicPDE{1}, 
    particleGrid::ParticleGrid1D, 
    settings::SimSetting, 
    time::Real, 
    dt::Real
)
    if !particleGrid.regular
        @warn "Classical timesteppers are designed for regular grids. Results may be inaccurate."
    end

    # Copy the initial state for all particles (including ghosts) into the buffer
    map!(particle -> particle.rho, cts_mood.rhoOld, particleGrid.grid)
    
    dx = particleGrid.dx
    dtdx = dt / dx
    
    interior_indices = particleGrid.interior_indices
    N_total = length(particleGrid.grid)

    # --- 1. Predictor Step (Lax-Wendroff): Calculate U_{i+1/2}^{n+1/2} ---
    # This is done for all interfaces, including those involving ghost cells.
    if particleGrid.bc == :periodic
        for i in 1:N_total
            idx_plus_1 = mod1(i + 1, N_total)
            Ui_n = cts_mood.rhoOld[i]; Uip1_n = cts_mood.rhoOld[idx_plus_1]
            F_Ui_n = flux(eq, Ui_n); F_Uip1_n = flux(eq, Uip1_n)
            cts_mood.rhoPredict_interface[i] = 0.5 * (Ui_n + Uip1_n) - (dt / (2.0 * dx)) * (F_Uip1_n - F_Ui_n)
        end
    else # Fixed BC
        for i in 1:(N_total - 1)
            Ui_n = cts_mood.rhoOld[i]; Uip1_n = cts_mood.rhoOld[i+1]
            F_Ui_n = flux(eq, Ui_n); F_Uip1_n = flux(eq, Uip1_n)
            cts_mood.rhoPredict_interface[i] = 0.5 * (Ui_n + Uip1_n) - (dt / (2.0 * dx)) * (F_Uip1_n - F_Ui_n)
        end
    end

    # --- 2. Corrector Step (Candidate Solution) ---
    # Calculate a high-order candidate solution for all INTERIOR cells.
    flux_of_predicted_states = [flux(eq, val) for val in cts_mood.rhoPredict_interface]

    for i in interior_indices
        F_star_ip_half = particleGrid.bc == :periodic ? flux_of_predicted_states[i] : flux_of_predicted_states[i]
        F_star_im_half = particleGrid.bc == :periodic ? flux_of_predicted_states[mod1(i - 1, N_total)] : flux_of_predicted_states[i - 1]
        
        cts_mood.rhoCandidate[i] = cts_mood.rhoOld[i] - dtdx * (F_star_ip_half - F_star_im_half)
    end

    # --- 3. MOOD Detection and Final Update ---
    # Loop through interior cells, check the candidate, and apply final update.
    for i in interior_indices
        # The mood function needs the full old state vector to find local extrema.
        if cts_mood.mood(particleGrid, i, cts_mood.rhoOld, cts_mood.rhoCandidate[i])
            # MOOD triggered! Recalculate update for this cell using Lax-Friedrichs fallback.
            
            # Get states for left and right interfaces of cell i
            u_L_left_interface = cts_mood.rhoOld[particleGrid.bc == :periodic ? mod1(i-1, N_total) : i-1]
            u_R_left_interface = cts_mood.rhoOld[i]
            
            u_L_right_interface = cts_mood.rhoOld[i]
            u_R_right_interface = cts_mood.rhoOld[particleGrid.bc == :periodic ? mod1(i+1, N_total) : i+1]
            
            # Lax-Friedrichs flux at i-1/2
            alpha_minus = _max_abs_speed_classical(eq, u_L_left_interface, u_R_left_interface)
            F_star_im_half_LF = 0.5 * (flux(eq, u_L_left_interface) + flux(eq, u_R_left_interface)) - 0.5 * alpha_minus * (u_R_left_interface - u_L_left_interface)

            # Lax-Friedrichs flux at i+1/2
            alpha_plus = _max_abs_speed_classical(eq, u_L_right_interface, u_R_right_interface)
            F_star_ip_half_LF = 0.5 * (flux(eq, u_L_right_interface) + flux(eq, u_R_right_interface)) - 0.5 * alpha_plus * (u_R_right_interface - u_L_right_interface)
            
            # Apply the low-order, stable update
            particleGrid.grid[i].rho = cts_mood.rhoOld[i] - dtdx * (F_star_ip_half_LF - F_star_im_half_LF)
        else
            # Candidate is good, accept it.
            particleGrid.grid[i].rho = cts_mood.rhoCandidate[i]
        end
    end
end
