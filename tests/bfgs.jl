load("src/init.jl")

function f(x)
  x[1]^2 + (2.0 - x[2])^2
end
function g(x)
  [2.0 * x[1], -2.0 * (2.0 - x[2])]
end

initial_x = [100.0, 100.0]
initial_h = eye(2)

results = bfgs(f, g, initial_x, initial_h, 10e-8)
@assert norm(results[1] - [0.0, 2.0]) < 0.01

results = bfgs(f, g, initial_x, initial_h, 10e-16)
@assert norm(results[1] - [0.0, 2.0]) < 0.01

results = bfgs(f, g, initial_x, initial_h, 10e-32)
@assert norm(results[1] - [0.0, 2.0]) < 0.01
