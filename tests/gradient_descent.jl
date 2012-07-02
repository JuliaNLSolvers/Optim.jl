load("src/init.jl")

function f(x)
  (x[1] - 5.0)^2
end

function g(x)
  [2.0 * (x[1] - 5.0)]
end

results = gradient_descent(f, g, [0.0], 0.1, 10e-8)
@assert norm(results.minimum - [5.0]) < 0.01

results = gradient_descent(f, g, [0.0], 0.1, 10e-16)
@assert norm(results.minimum - [5.0]) < 0.01

results = gradient_descent(f, g, [0.0], 0.1, 10e-32)
@assert norm(results.minimum - [5.0]) < 0.01

eta = 0.9

function f(x)
  (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g(x)
  [x[1], eta * x[2]]
end

results = gradient_descent(f, g, [1, 1], 0.1, 10e-8)
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
