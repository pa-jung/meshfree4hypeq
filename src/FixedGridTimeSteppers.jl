export Upwind, LaxFriedrich, ClassicalTimeStepper, ClassicalRK2LWTimeStepper
using ..Meshfree4ScalarEq.FluxFunctions

struct Upwind <: FixedGridTimeStepper 
    rhoOld::Vector{Float64}
    function Upwind(Nx::Integer)
        new(Vector{Float64}(undef, Nx))
    end
end

function (upwind::Upwind)(eq::LinearAdvection, particleGrid::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
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

function (upwind::Upwind)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    error("Upwind method for nonlinear hyperbolic equations (Roe's scheme) not yet implemented.")
end

struct LaxFriedrich <: FixedGridTimeStepper 
    rhoOld::Vector{Float64}
    function LaxFriedrich(Nx::Integer)
        new(Vector{Float64}(undef, Nx))
    end
end

function (lf::LaxFriedrich)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
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

function (cts::ClassicalTimeStepper)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    map!(particle -> particle.rho, cts.rhoOld, particleGrid.grid)
    N = particleGrid.N
    dx = particleGrid.dx
    dtdx = dt / dx

    # Temporary arrays for interface fluxes F*_{i-1/2} and F*_{i+1/2} for each cell i
    # Or, calculate them on the fly. For clarity, let's pre-calculate all F*_{k+1/2}
    
    # flux_at_interfaces will store F*_{k+1/2} at index k
    # So, flux_at_interfaces[i] is F*_{i+1/2}
    # and flux_at_interfaces[mod1(i-1,N)] is F*_{i-1/2}
    flux_at_interfaces = Vector{Float64}(undef, N)

    # Calculate all interface fluxes F*_{k+1/2}
    # The interface k+1/2 is between cell k and cell k+1
    for k in 1:N
        u_L = cts.rhoOld[k]
        u_R = cts.rhoOld[mod1(k + 1, N)] # Periodic neighbor for U_R
        
        # Call the provided numerical flux function (e.g., RusanovFlux, UpwindFlux, LaxWendroffFlux)
        flux_at_interfaces[k] = cts.numericalFlux(u_L, u_R, eq)
    end

    # Update rule: U_i^{n+1} = U_i^n - (dt/dx) * ( F*_{i+1/2} - F*_{i-1/2} )
    for i in 1:N
        F_star_i_plus_half = flux_at_interfaces[i] # Flux at the right interface of cell i
        F_star_i_minus_half = flux_at_interfaces[mod1(i - 1, N)] # Flux at the left interface of cell i
        
        particleGrid.grid[i].rho = cts.rhoOld[i] - dtdx * (F_star_i_plus_half - F_star_i_minus_half)
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

function (crk2::ClassicalRK2LWTimeStepper)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
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
