load("src/optim.jl")

f = x -> x[1]^2 + x[2]^2

a = 1.0
g = 0.9
b = 1.1

initial_p = [0. 0.; 0. 1.; 1. 0.;]

tolerance = 10e-7

max_iterations = 100

solution = nelder_mead(f, initial_p, a, g, b, tolerance, max_iterations)

@assert norm(solution[1] - [0 0]) < 0.01

initial_p = [-10. -15.; 5. 1.; 1. 17.;]

tolerance = 10e-16

max_iterations = 10000

solution = nelder_mead(f, initial_p, a, g, b, tolerance, max_iterations)

# This fails. Why? Algorithm or code?
