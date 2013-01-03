function f2(x)
  x[1]^2 + (2.0 - x[2])^2
end
function g2(x)
  [2.0 * x[1], -2.0 * (2.0 - x[2])]
end

initial_x = [100.0, 100.0]
initial_h = eye(2)

results = Optim.bfgs(f2, g2, initial_x, initial_h, 10e-8, 1_000, false, false)
@assert length(results.trace.states) == 0
@assert results.converged
@assert norm(results.minimum - [0.0, 2.0]) < 0.01

results = Optim.bfgs(f2, g2, initial_x, initial_h, 10e-8, 1_000, true, false)
@assert length(results.trace.states) > 0
@assert results.converged
@assert norm(results.minimum - [0.0, 2.0]) < 0.01

results = Optim.bfgs(f2, g2, initial_x, initial_h)
@assert norm(results.minimum - [0.0, 2.0]) < 0.01
