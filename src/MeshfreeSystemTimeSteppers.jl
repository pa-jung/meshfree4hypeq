
# Default case: Everything is decoupled

function (ts::TimeStepper)(eqs::Vector{<:ScalarHyperbolicEquation}, particleGrids::Vector{<:ParticleGrid}, settings::SimSetting, time::Real, dt::Real)
    @warn "No dedicated system timestepper found. System will be treated independently using the given scalar timestepper."
    for (i,particleGrid) = enumerate(particleGrids)
        ts(eqs[i], particleGrid, settings, time, dt)
    end
end

function (ts::TimeStepper)(eq:ScalarHyperbolicEquation, particleGrids::Vector{<:ParticleGrid}, settings::SimSetting, time::Real, dt::Real)
    @warn "Only one scalar hyperbolic equation for a system is found. The scalar equation will be used for all components!"
    ts([eq for _ = eachindex(particleGrids)], particleGrids, settings, time, dt)
end