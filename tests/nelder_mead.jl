load("src/init.jl")

function f(x)
  x[1]^2 + x[2]^2
end

a = 1.0
g = 2.0
b = 0.5

initial_p = [0.0 0.0; 0.0 1.0; 1.0 0.0;]
tolerance = 10e-8
max_iterations = 100

results = nelder_mead(f, initial_p, a, g, b, tolerance, max_iterations, false)
@assert norm(results[1] - [0.0 0.0]) < 0.01

initial_p = [-10.0 -15.0; 5.0 1.0; 1.0 17.0;]
tolerance = 10e-16
max_iterations = 100

results = nelder_mead(f, initial_p, a, g, b, tolerance, max_iterations, true)
@assert norm(results[1] - [0.0 0.0]) < 0.01

function rosenbrock(x)
  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

initial_p = [-10.0 -15.0; 5.0 1.0; 1.0 17.0;]
tolerance = 10e-8
max_iterations = 1000

results = nelder_mead(rosenbrock, initial_p, a, g, b, tolerance, max_iterations, false)
@assert norm(results[1] - [1.0 1.0]) < 0.01
