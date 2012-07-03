load("src/init.jl")

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

results = optimize(f, g, h, [127.0, 921.0])
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

results = optimize(f, g, [127.0, 921.0])
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

results = optimize(f, [127.0, 921.0])
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
