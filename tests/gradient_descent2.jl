load("src/init.jl")

function f(x)
  (x[1] - 5.0)^2
end

function g(x)
  [2.0 * (x[1] - 5.0)]
end

results = gradient_descent2(f, g, [0.0], 10e-8, 0.1, 0.8)
@assert norm(results.minimum - [5.0]) < 0.01

results = gradient_descent2(f, g, [0.0], 10e-16, 0.1, 0.8)
@assert norm(results.minimum - [5.0]) < 0.01

results = gradient_descent2(f, g, [0.0], 10e-32, 0.1, 0.8)
@assert norm(results.minimum - [5.0]) < 0.01

eta = 0.9

function f(x)
  (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g(x)
  [x[1], eta * x[2]]
end

results = gradient_descent2(f, g, [1.0, 1.0], 10e-8, 0.1, 0.8)
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
