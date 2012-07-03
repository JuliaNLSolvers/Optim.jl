load("src/init.jl")

function f(x)
  (x[1] - 5.0)^4
end

function g(x)
  [4.0 * (x[1] - 5.0)^3]
end

function h(x)
  a = zeros(1, 1)
  a[1, 1] = 12.0 * (x[1] - 5.0)^2
  a
end

results = newton(f, g, h, [0.0], 10e-16, 1000, true)
@assert norm(results.minimum - 5.0) < 0.01

results = newton(f, g, h, [0.0])
@assert norm(results.minimum - 5.0) < 0.01

eta = 0.9

function f(x)
  (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g(x)
  [x[1], eta * x[2]]
end

function h(x)
  [1.0 0.0; 0.0 eta]
end

results = newton(f, g, h, [127.0, 921.0], 10e-16, 1000, true)
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

results = newton(f, g, h, [127.0, 921.0])
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
