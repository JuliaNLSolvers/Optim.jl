eta = 0.9

function f1(x)
    (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g1(x, storage)
    storage[1] = x[1]
    storage[2] = eta * x[2]
end

function h1(x, storage)
    storage[1, 1] = 1.0
    storage[1, 2] = 0.0
    storage[2, 1] = 0.0
    storage[2, 2] = eta
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
