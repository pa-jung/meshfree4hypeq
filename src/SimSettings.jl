module SimSettings

export SimSetting

"""
    SimSetting

Struct that stores some simulation parameters together in one place.

# Arguments:
- `tmax::Float64`: Final simulation time.
- `dt::Float64`: Time step.
- `interpRange::Float64`: Interpolation range.
- `interpAlpha::Float64`: Alpha parameter for weight function in interpolation routines.
"""
struct SimSetting
    tmax::Float64
    dt::Float64
    interpRange::Float64
    interpAlpha::Float64
    saveFreq::UInt64

    function SimSetting(; tmax::Real, dt::Real, interpRange::Real, interpAlpha::Real, saveFreq::Integer)
        new(convert(Float64, tmax), convert(Float64, dt), convert(Float64, interpRange), convert(Float64, interpAlpha), convert(UInt64, saveFreq))
    end
    function SimSetting(tmax::Real, dt::Real, interpRange::Real, interpAlpha::Real, saveFreq::Real)
        new(convert(Float64, tmax), convert(Float64, dt), convert(Float64, interpRange), convert(Float64, interpAlpha), convert(UInt64, saveFreq))
    end
end

end  # module SimSettings