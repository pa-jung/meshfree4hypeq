module FluxFunctions

using ..Meshfree4ScalarEq.ScalarHyperbolicEquations

export NumericalFluxFunction, RusanovFlux, UpwindFlux, LaxFriedrichsFlux, LaxWendroffFlux

abstract type NumericalFluxFunction end;

# ---------- RusanovFlux (LLF)
struct RusanovFlux <: NumericalFluxFunction end

function (rusanov::RusanovFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicEquation)::Real
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    s = max(abs(velocity(eq, leftState)), abs(velocity(eq, rightState)))
    return 0.5*(leftFlux + rightFlux - s*(rightState - leftState))
end

function (rusanov::RusanovFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicEquation, ind::Int)::Real  # In case of 2D, select correct velocity (1 for x, 2 for y)
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState) 
    a_l = velocity(eq, leftState)
    a_r = velocity(eq, rightState)
    s = max(abs(a_l[ind]), abs(a_r[ind]))
    return 0.5*(leftFlux[ind] + rightFlux[ind] - s*(rightState - leftState))
end

# ---------- UpwindFlux
struct UpwindFlux <: NumericalFluxFunction end

function (upwind::UpwindFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicEquation)  
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    a = leftState == rightState ?  velocity(eq, leftState) : (leftFlux - rightFlux)/(leftState - rightState)
    return 0.5*(leftFlux + rightFlux - abs(a)*(rightState - leftState))
end

function (upwind::UpwindFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicEquation, ind::Int)  
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    a = leftState == rightState ?  velocity(eq, leftState)[ind] : (leftFlux[ind] - rightFlux[ind])/(leftState - rightState)
    return 0.5*(leftFlux[ind] + rightFlux[ind] - abs(a)*(rightState - leftState))
end

#--------------- LaxFriedrichFlux
struct LaxFriedrichsFlux <: NumericalFluxFunction end

function (lf::LaxFriedrichsFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicEquation, alpha::Real)::Real
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    return 0.5*(leftFlux + rightFlux - alpha*(rightState - leftState))
end

#--------------- LaxWendroffFlux
struct LaxWendroffFlux <: NumericalFluxFunction end

function (lw::LaxWendroffFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicEquation)::Real
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    s = max(abs(velocity(eq, leftState)), abs(velocity(eq, rightState)))
    return 0.5*(leftFlux + rightFlux - s^2*(rightState - leftState))
end

end