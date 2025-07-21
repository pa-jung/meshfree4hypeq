using JLD2
using Plots
using VoronoiCells
using GeometryBasics
using Meshfree4ScalarEq.ParticleGrids
plotlyjs()

function getMassLossData(Nx::Integer)
    dataFolder = "$(@__DIR__)/../L2Stability2D/data/"
    algList = Vector{String}(undef, 0)
    fileList = Vector{String}(undef, 0)
    for file in readdir(dataFolder)
        if contains(file, "$(Nx)") && (file[end-2:end] != "txt") && (file[end-2:end] != "pdf")
            alg = split(file, "_")[1]
            push!(algList, alg)
            push!(fileList, dataFolder*file)
        end
    end

    D = Dict{String, Matrix{Float64}}()
    for (j, file) in enumerate(fileList)
        steps = count((contains(t, "step" ) for t in readdir(file * "/data/")))
        data = Matrix{Float64}(undef, steps, 2)
        i = 1
        for dataFile in readdir(file * "/data/")
            if contains(dataFile, "step")
                d = load(file * "/data/" * dataFile)
                data[i, 1] = d["time"]
                data[i, 2] = sum(particle ->  particle.rho*particle.volume, d["particleGrid"].grid)
                i += 1
            end
        end
        # println(algList[j], " ", data)

        # Sort increase time
        perm = sortperm(data[:, 1])
        data[:, 1] .= data[perm, 1]
        data[:, 2] .= data[perm, 2]

        # Normalise 
        data[:, 2] .= data[:, 2]/data[1, 2]

        # Save to dict
        D[algList[j]] = data
    end
    return D, algList
end

D100, algList100 = getMassLossData(100);
D80, algList80 = getMassLossData(80);

ms = 3
p1 = plot(D100[algList100[2]][:, 1], D100[algList100[2]][:, 2], xlabel="Time", ylabel="Normalised mass", ylim=(0.8, 1.5), label="RK3 + MUSCL", markershape=:circle, size=(800, 600), legend=:topright, ms=ms)
plot!(p1, D100[algList100[6]][:, 1], D100[algList100[6]][:, 2], label="RK3 + WENO", markershape=:circle, ms=ms)
savefig(p1, "$(@__DIR__)/figures/massLoss100.pdf")
display(p1)

p2 = plot(D80[algList80[2]][:, 1], D80[algList80[2]][:, 2], xlabel="Time", ylabel="Normalised mass", ylim=(0.8, 2.0), label="RK3 + MUSCL", markershape=:circle, size=(800, 600), legend=:topright, ms=ms)
plot!(p2, D80[algList80[6]][:, 1], D80[algList80[6]][:, 2], label="RK3 + WENO", markershape=:circle, ms=ms)
savefig(p2, "$(@__DIR__)/figures/massLoss80.pdf")
display(p2)