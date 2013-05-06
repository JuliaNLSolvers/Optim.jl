function f_gd(x)
  (x[1] - 5.0)^2
end

function g_gd(x, storage)
  storage[1] = 2.0 * (x[1] - 5.0)
end

initial_x = [0.0]

d = DifferentiableFunction(f_gd, g_gd)

results = Optim.gradient_descent(d, initial_x)
@assert isempty(results.trace.states)
@assert results.gr_converged
@assert norm(results.minimum - [5.0]) < 0.01

eta = 0.9

function f_gd(x)
  (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g_gd(x, storage)
  storage[1] = x[1]
  storage[2] = eta * x[2]
end

d = DifferentiableFunction(f_gd, g_gd)

results = Optim.gradient_descent(d, [1.0, 1.0])
@assert isempty(results.trace.states)
@assert results.gr_converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
