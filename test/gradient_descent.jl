function f_gd_1(x)
  (x[1] - 5.0)^2
end

function g_gd_1(x, storage)
  storage[1] = 2.0 * (x[1] - 5.0)
end

initial_x = [0.0]

d = DifferentiableFunction(f_gd_1, g_gd_1)

results = Optim.optimize(d, initial_x, method=GradientDescent())
@assert isempty(results.trace.states)
@assert results.gr_converged
@assert norm(results.minimum - [5.0]) < 0.01

eta = 0.9

function f_gd_2(x)
  (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g_gd_2(x, storage)
  storage[1] = x[1]
  storage[2] = eta * x[2]
end

d = DifferentiableFunction(f_gd_2, g_gd_2)

results = Optim.optimize(d, [1.0, 1.0], method=GradientDescent())
@assert isempty(results.trace.states)
@assert results.gr_converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
