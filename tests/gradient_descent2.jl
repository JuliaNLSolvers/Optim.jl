load("src/optim.jl")

f = x -> (x - 5) ^ 2
g = x -> 2(x - 5)

@assert abs(gradient_descent2(f, g, 0, 10e-8, 0.1, 0.8)[1] - 5) < 0.01
@assert abs(gradient_descent2(f, g, 0, 10e-16, 0.1, 0.8)[1] - 5) < 0.01
@assert abs(gradient_descent2(f, g, 0, 10e-32, 0.1, 0.8)[1] - 5) < 0.01

eta = 0.9

f = x -> (1/2) * (x[1]^2 + eta * x[2]^2)
g = x -> [x[1], eta * x[2]]

@assert norm(gradient_descent2(f, g, [1, 1], 10e-8, 0.1, 0.8)[1] - [0, 0]) < 0.01
