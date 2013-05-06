function f(x::Vector)
  (100.0 - x[1])^2 + x[2]^2
end

function rosenbrock(x::Vector)
  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

initial_x = [0.0, 0.0]

results = Optim.nelder_mead(f, initial_x)

@assert results.f_converged
@assert norm(results.minimum - [100.0, 0.0]) < 0.01
@assert length(results.trace.states) == 0

results = Optim.nelder_mead(rosenbrock, initial_x)

@assert results.f_converged
@assert norm(results.minimum - [1.0, 1.0]) < 0.01
@assert length(results.trace.states) == 0
