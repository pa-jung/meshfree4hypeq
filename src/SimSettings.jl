module SimSettings

using FileIO, JLD2

export SimSetting, saveSettings

"""
    SimSetting

Struct that stores some simulation parameters together in one place.

# Arguments:
- `tmax::Float64`: Final simulation time.
- `dt::Float64`: Time step.
- `interpRange::Float64`: Interpolation range.
- `interpAlpha::Float64`: Alpha parameter for weight function in interpolation routines.
"""
mutable struct SimSetting
    const tmax::Float64
    const dt::Float64
    const interpRange::Float64
    const interpAlpha::Float64
    const saveDir::String
    const saveFreq::UInt64
    currentSaveNb::UInt64

    function SimSetting(; tmax::Real, dt::Real, interpRange::Real, interpAlpha::Real, saveDir::String, saveFreq::Integer, organiseFiles::Bool=true)
        @assert endswith(saveDir, "/")
        if organiseFiles
            _generateDir(saveDir)
            _cleanDir(saveDir)
        end
        new(convert(Float64, tmax), convert(Float64, dt), convert(Float64, interpRange), convert(Float64, interpAlpha), saveDir, convert(UInt64, saveFreq), UInt64(0))
    end
    function SimSetting(;tmax::Real, dt::Real, interpRange::Real, interpAlpha::Real, saveFreq::Real)
        new(convert(Float64, tmax), convert(Float64, dt), convert(Float64, interpRange), convert(Float64, interpAlpha), "/", convert(UInt64, saveFreq), UInt64(0))
    end
end

"""
    _generateDir(dir::String)

Make data/ and figures/ subfolders if they don't already exist.
"""
function _generateDir(dir::String)
    if !isdir(dir)
        mkdir(dir)
    end
    if !isdir(dir*"data/")
        mkdir(dir*"data/")
    end
    if !isdir(dir*"figures/")
        mkdir(dir*"figures/")
    end
end

"""
    saveSettings(settings::SimSetting)

Store settings object.
"""
function saveSettings(settings::SimSetting)
    save("$(settings.saveDir)data/settings.jld2", "settings", settings)
end

"""
    cleanDir(dir::String)

Remove all .jld2 files in simDir/data/ and all .png and .pdf files in simDir/figures/.
"""
function _cleanDir(dir::String)
    for file in filter(x->endswith(x, ".jld2"), readdir("$(dir)data/"))
        rm(dir*"data/"*file)
    end
    for imageType in [".png", ".pdf", ".gif"]
        for file in filter(x-> endswith(x, imageType), readdir("$(dir)figures/"))
            rm(dir*"figures/"*file)
        end
    end
end

end  # module SimSettings