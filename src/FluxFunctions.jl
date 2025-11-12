module FluxFunctions

using ..Meshfree4ScalarEq.HyperbolicPDEs

export NumericalFluxFunction, RusanovFlux, UpwindFlux, RoeDiffusiveFlux

abstract type NumericalFluxFunction end;

# ---------- RusanovFlux (LLF)
struct RusanovFlux <: NumericalFluxFunction end

function (rusanov::RusanovFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicPDE{D})::Real where {D}
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    s = max(abs(velocity(eq, leftState)), abs(velocity(eq, rightState)))
    return 0.5*(leftFlux + rightFlux - s*(rightState - leftState))
end

function (rusanov::RusanovFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicPDE{D}, ind::Int)::Real where {D} # In case of 2D, select correct velocity (1 for x, 2 for y)
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState) 
    a_l = velocity(eq, leftState)
    a_r = velocity(eq, rightState)
    s = max(abs(a_l[ind]), abs(a_r[ind]))
    return 0.5*(leftFlux[ind] + rightFlux[ind] - s*(rightState - leftState))
end

# ---------- UpwindFlux
struct UpwindFlux <: NumericalFluxFunction end

function (upwind::UpwindFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicPDE{D}) where {D} 
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    a = leftState == rightState ?  velocity(eq, leftState) : (leftFlux - rightFlux)/(leftState - rightState)
    return 0.5*(leftFlux + rightFlux - abs(a)*(rightState - leftState))
end

function (upwind::UpwindFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicPDE{D}, ind::Int) where {D} 
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    a = leftState == rightState ?  velocity(eq, leftState)[ind] : (leftFlux[ind] - rightFlux[ind])/(leftState - rightState)
    return 0.5*(leftFlux[ind] + rightFlux[ind] - abs(a)*(rightState - leftState))
end

#--------------- RoeDiffusiveFlux (Lax Wendroff without λ scaling)
struct RoeDiffusiveFlux <: NumericalFluxFunction end

function (lw::RoeDiffusiveFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicPDE{D})::Real where {D}
    F_L = flux(eq, leftState)
    F_R = flux(eq, rightState)
    
    avg_F = 0.5 * (F_L + F_R)
    diff_U = rightState - leftState

    if abs(diff_U) < 1e-12
        return F_L # or F_R, they are the same
    else
        A_roe_squared_term = (F_L - F_R)^2 / diff_U # This is (-(F_R-F_L))^2 / diff_U = (F_R-F_L)^2 / diff_U
        return avg_F - A_roe_squared_term
    end
end

end