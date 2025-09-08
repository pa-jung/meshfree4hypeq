module HyperbolicPDEs

export ScalarHyperbolicEquation, LinearScalarHyperbolicEquation, NonLinearScalarHyperbolicEquation, LinearAdvection, 
       BurgersEquation, BurgersEquation2D, velocity, flux, HyperbolicSystem, Euler1D, Euler2D, pressure_from_euler_conserved

abstract type HyperbolicPDE end


include("ScalarHyperbolicEquations.jl")
include("HyperbolicSystems.jl")

end