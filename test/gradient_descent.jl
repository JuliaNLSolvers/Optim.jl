function f_gd(x)
  (x[1] - 5.0)^2
end

function g_gd(x)
  [2.0 * (x[1] - 5.0)]
end

initial_x = [0.0]

results = Optim.gradient_descent(f_gd, g_gd, initial_x)
@assert isempty(results.trace.states)
@assert results.converged
@assert norm(results.minimum - [5.0]) < 0.01

eta = 0.9

function f_gd(x)
  (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g_gd(x)
  [x[1], eta * x[2]]
end

results = Optim.gradient_descent(f_gd, g_gd, [1.0, 1.0])
@assert isempty(results.trace.states)
@assert results.converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
