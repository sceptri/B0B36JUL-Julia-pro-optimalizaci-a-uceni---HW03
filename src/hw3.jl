# IN SOME FUNCTIONS, IT MAY BE NECESSARY TO ADD KEYWORD ARGUMENTS

using RDatasets
using LinearAlgebra

computeQ(X, y) = (y .* X) * (y .* X)'
computeW(X, y, z) = sum(y[i] * z[i] .* row for (i, row) in enumerate(eachrow(X)))

φ(z, Q) = -0.5 * (z'*Q*z)[1] + sum(z)
φᵢ(d, z, Q, eᵢ) = φ(z + d .* eᵢ, Q)
dφᵢ(d, z, Q, eᵢ) = -d .* eᵢ' * Q * eᵢ .- 0.5 .* eᵢ' * Q * z .- 0.5 .* z' * Q * eᵢ .+ 1

f(w, X, y, C) = C * sum(max(0, 1 - y[i] * (w'*x)[1]) for (i, x) in enumerate(eachrow(X))) + 0.5 * norm(w)^2

function solve_SVM_dual(Q, C; max_epoch=2000, kwargs...)
    samples = size(Q, 1)
    z = zeros(Float64, samples)

    for _ in 1:max_epoch, i in 1:samples
        eᵢ = zeros(Float64, samples)
        eᵢ[i] = 1

        # Got by hand from the derivative dφᵢ
        d = (1 - 0.5 * (eᵢ'*Q*z+z'*Q*eᵢ)[1]) / (eᵢ'*Q*eᵢ)[1]

        bounds = [-z[i], C - z[i]]
        candidates = bounds
        if bounds[1] <= d <= bounds[2]
            push!(candidates, d)
        end
        ϕ(d) = φᵢ(d, z, Q, eᵢ)

        d = candidates[argmax(ϕ.(candidates))]
        z += d .* eᵢ
    end

    return z
end

function solve_SVM(X, y, C; kwargs...)
    Q = computeQ(X, y)
    z = solve_SVM_dual(Q, C; kwargs...)
    w = computeW(X, y, z)
    return w
end

function iris(C; kwargs...)
    data = dataset("datasets", "iris")
    data_reduced = data[data.Species.!="setosa", :]
    X = Matrix(data_reduced[:, 3:4])
    X = hcat(X, ones(Float64, size(X, 1)))
    y = Int.(data_reduced.Species .== "versicolor") .* 2 .- 1
    w = solve_SVM(X, y, C; kwargs...)
    return w
end

