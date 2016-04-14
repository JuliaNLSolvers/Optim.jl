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

for method in (NelderMead(),
               SimulatedAnnealing())
    optimize(rosenbrock, [0.0,0,.0], method = method)
    optimize(rosenbrock, Float32[0.0, 0.0], method = method)
end

for method in (BFGS(),
               ConjugateGradient(),
               GradientDescent(),
               MomentumGradientDescent(),
#                :accelerated_gradient_descent,
               LBFGS())
    optimize(d2, [0.0,0,.0], method = method)
    optimize(d2, Float32[0.0, 0.0], method = method)
end

for method in (Newton(),)
    optimize(d3, [0.0,0.0], method = method)
    optimize(d3, Float32[0.0, 0.0], method = method)
end
