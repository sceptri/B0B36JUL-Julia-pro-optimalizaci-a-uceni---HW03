using Test

include("../src/hw3.jl")

data = dataset("datasets", "iris")
data_reduced = data[data.Species.!="setosa", :]

C = 1 / 100
X = Matrix(data_reduced[:, 3:4])
X = hcat(X, ones(Float64, size(X, 1)))
y = Int.(data_reduced.Species .== "versicolor") .* 2 .- 1
Q = computeQ(X, y)
z = solve_SVM_dual(Q, C; max_epoch=10000)
w = computeW(X, y, z)

# Test that the values of primal and dual problems match
@test φ(z, Q) ≈ f(w, X, y, C)

# Visual Test
separ(x::Real, ω) = (-ω[3] - ω[1] * x) / ω[2]

Xlims = extrema(data_reduced.PetalLength) .+ [-0.1, 0.1]
Ylims = extrema(data_reduced.PetalWidth) .+ [-0.1, 0.1]

ω = iris(100; max_epoch=10000)
scatter(
    data_reduced.PetalLength,
    data_reduced.PetalWidth;
    group=data_reduced.Species,
    xlabel="Petal length",
    ylabel="Petal width",
    legend=:topleft,
    xlims=Xlims,
    ylims=Ylims
)

plot!(Xlims, x -> separ(x, ω); label="Separation", line=(:black, 3))