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
@assert results.gr_converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

results = optimize(f1, g1, [127.0, 921.0])
@assert results.gr_converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

results = optimize(f1, [127.0, 921.0])
@assert results.f_converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

results = optimize(f1, [127.0, 921.0], autodiff = true)
@assert results.f_converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

# tests for bfgs_initial_invH
initial_invH = zeros(2,2)
h1([127.0, 921.0],initial_invH)
initial_invH = diagm(diag(initial_invH))
results = optimize(DifferentiableFunction(f1, g1), [127.0, 921.0], BFGS(), OptimizationOptions(),
                   initial_invH = initial_invH)
@assert results.gr_converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
