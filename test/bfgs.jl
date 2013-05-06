function f2(x)
  x[1]^2 + (2.0 - x[2])^2
end
function g2(x, storage)
  storage[1] = 2.0 * x[1]
  storage[2] = -2.0 * (2.0 - x[2])
end
d2 = DifferentiableFunction(f2, g2)
initial_x = [100.0, 100.0]

results = Optim.bfgs(d2, initial_x)
@assert length(results.trace.states) == 0
@assert results.gr_converged
@assert norm(results.minimum - [0.0, 2.0]) < 0.01
