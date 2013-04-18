function f(x::Vector)
  (309.0 - 5.0 * x[1])^2 + (17.0 - x[2])^2
end
function g!(x::Vector, storage::Vector)
  storage[1] = -10.0 * (309.0 - 5.0 * x[1])
  storage[2] = -2.0 * (17.0 - x[2])
end

d = DifferentiableFunction(f, g!)

initial_x = [10.0, 10.0]
m = 10
store_trace, show_trace = false, false

results = Optim.l_bfgs(d, initial_x)
@assert length(results.trace.states) == 0
@assert results.converged
@assert norm(results.minimum - [309.0 / 5.0, 17.0]) < 0.01
