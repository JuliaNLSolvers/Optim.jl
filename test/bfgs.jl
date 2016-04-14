function f2(x)
  x[1]^2 + (2.0 - x[2])^2
end
function g2(x, storage)
  storage[1] = 2.0 * x[1]
  storage[2] = -2.0 * (2.0 - x[2])
end
d2 = DifferentiableFunction(f2, g2)
initial_x = [100.0, 100.0]

results = Optim.optimize(d2, initial_x, method=BFGS())
@test_throws ErrorException Optim.trace(results)
@assert Optim.g_converged(results)
@assert norm(Optim.minimizer(results) - [0.0, 2.0]) < 0.01

d2 = Optim.autodiff(f2, Float64, 2)
results = Optim.optimize(d2, initial_x, method=BFGS())
@test_throws ErrorException Optim.trace(results)
@assert Optim.g_converged(results)
@assert norm(Optim.minimizer(results) - [0.0, 2.0]) < 0.01
