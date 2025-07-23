module HyperbolicSystems

export HyperbolicSystem, EulerEquations

abstract type HyperbolicSystem end

struct EulerEquations <: HyperbolicSystem end

end