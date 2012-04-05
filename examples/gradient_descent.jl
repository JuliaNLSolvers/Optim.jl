load("src/optim.jl")

f = x -> (x - 5) ^ 2
g = x -> 2(x - 5)

@assert abs(gradient_descent(f, g, 0, 0.1, 10e-8)[1] - 5) < 0.01
@assert abs(gradient_descent(f, g, 0, 0.1, 10e-16)[1] - 5) < 0.01
@assert abs(gradient_descent(f, g, 0, 0.1, 10e-32)[1] - 5) < 0.01
