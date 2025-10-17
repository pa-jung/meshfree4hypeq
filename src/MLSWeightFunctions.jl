module MLSWeightFunctions

export MLSWeightFunction, exponentialWeightFunction, inverseWeightFunction

# --- Abstract Type ---
abstract type MLSWeightFunction end

# --- Struct Definitions with Parameters ---

"""
    exponentialWeightFunction(alpha::Real, range::Real)

Functor that calculates an exponential weight based on distance.
The parameters `alpha` (shape parameter) and `range` (normalization distance)
are stored directly in the struct.
"""
struct exponentialWeightFunction <: MLSWeightFunction
    alpha::Float64
    range::Float64
    inv_range_sq::Float64
    function exponentialWeightFunction(alpha::Float64, range::Float64)
        inv_range_sq = 1.0 / (w.range^2)
        new(alpha,range,inv_range_sq)
    end
end

"""
    inverseWeightFunction(alpha::Real=0.0, range::Real=0.0)

Functor that calculates an inverse-square distance weight.
The parameters `alpha` and `range` are included for a consistent
interface but are not used in the calculation.
"""
struct inverseWeightFunction <: MLSWeightFunction
    alpha::Float64
    range::Float64
end
# Provide a default constructor
inverseWeightFunction() = inverseWeightFunction(0.0, 0.0)


# --- Fast Exponential Approximations (unchanged) ---

"""
A fast, high-accuracy, and stable approximation of `exp(x)` for `x <= 0`.
"""
@inline function fast_exp_accurate(x::Float64)
    y = -x
    denominator = 1.0 + y * (1.0 + y * (0.5 + y * (0.16666666666666666 + y * 0.041666666666666664)))
    return 1.0 / denominator
end


# --- SCALAR Functor Implementations ---

"""
Calculates the exponential weight for a single interaction given the squared distance.
This is the most efficient version for use inside loops.
"""
@inline function (w::exponentialWeightFunction)(dist_sq::Real)
    arg = -w.alpha * dist_sq * w.inv_range_sq
    return fast_exp_accurate(arg)
end

"""
Calculates the inverse-square weight for a single interaction given the squared distance.
"""
@inline function (w::inverseWeightFunction)(dist_sq::Real)
    # Add a small epsilon to prevent division by zero if two points are identical
    return 1.0 / (dist_sq + 1e-12)
end

end # end of module