eta = 0.9

function f1(x)
    (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g1(x)
    [x[1], eta * x[2]]
end

function h1(x)
    [1.0 0.0; 0.0 eta]
end

results = optimize(f1, g1, h1, [127.0, 921.0])
@assert results.converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

results = optimize(f1, g1, [127.0, 921.0])
@assert results.converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

results = optimize(f1, [127.0, 921.0])
@assert results.converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
