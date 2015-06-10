function rosenbrock{T}(x::Vector{T})
    o = one(T)
    c = convert(T,100)
    return (o - x[1])^2 + c * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient!{T}(x::Vector{T}, storage::Vector{T})
    o = one(T)
    c = convert(T,100)
    storage[1] = (-2*o) * (o - x[1]) - (4*c) * (x[2] - x[1]^2) * x[1]
    storage[2] = (2*c) * (x[2] - x[1]^2)
end

function rosenbrock_hessian!{T}(x::Vector{T}, storage::Matrix{T})
    o = one(T)
    c = convert(T,100)
    f = 4*c
    storage[1, 1] = (2*o) - f * x[2] + 3 * f * x[1]^2
    storage[1, 2] = -f * x[1]
    storage[2, 1] = -f * x[1]
    storage[2, 2] = 2*c
end

d2 = DifferentiableFunction(rosenbrock,
                            rosenbrock_gradient!)
d3 = TwiceDifferentiableFunction(rosenbrock,
                                 rosenbrock_gradient!,
                                 rosenbrock_hessian!)

for method in (:nelder_mead,
               :simulated_annealing)
    optimize(rosenbrock, zeros(Float64,2), method = method)
    optimize(rosenbrock, zeros(Float32,2), method = method)
end

for method in (:bfgs,
               :cg,
               :gradient_descent,
               :momentum_gradient_descent,
#                :accelerated_gradient_descent,
               :l_bfgs)
    optimize(d2, zeros(Float64,2), method = method)
    optimize(d2, zeros(Float32,2), method = method)
end

for method in (:newton,)
    optimize(d3, zeros(Float64,2), method = method)
    optimize(d3, zeros(Float32,2), method = method)
end
