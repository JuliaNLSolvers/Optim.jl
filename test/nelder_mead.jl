function f(x::Vector)
  (100.0 - x[1])^2 + x[2]^2
end

function rosenbrock(x::Vector)
  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

(nm_a, nm_g, nm_b) = (1.0, 2.0, 0.5)
initial_p = [0.0 0.0; 0.0 1.0; 1.0 0.0;]' #'
tolerance = 10e-8
max_iterations = 100
store_trace, show_trace = false, false
results = Optim.nelder_mead(f,
	                        initial_p,
                            nm_a,
                            nm_g,
                            nm_b,
                            tolerance,
                            max_iterations,
                            store_trace,
                            show_trace)
@assert results.converged
@assert norm(results.minimum - [100.0, 0.0]) < 0.01
@assert length(results.trace.states) == 0

store_trace, show_trace = true, false
results = Optim.nelder_mead(f,
	                        initial_p,
                            nm_a,
                            nm_g,
                            nm_b,
                            tolerance,
                            max_iterations,
                            store_trace,
                            show_trace)
@assert results.converged
@assert norm(results.minimum - [100.0, 0.0]) < 0.01
@assert length(results.trace.states) > 0

initial_p = [-10.0 -15.0; 5.0 1.0; 1.0 17.0;]' #'
tolerance = 10e-16
max_iterations = 100
results = Optim.nelder_mead(f,
	                        initial_p,
                            nm_a,
                            nm_g,
                            nm_b,
                            tolerance,
                            max_iterations,
                            store_trace,
                            show_trace)
@assert results.converged
@assert norm(results.minimum - [100.0, 0.0]) < 0.01

initial_p = [-10.0 -15.0; 5.0 1.0; 1.0 17.0;]' #'
tolerance = 10e-8
max_iterations = 1_000
results = Optim.nelder_mead(rosenbrock,
	                        initial_p,
                            nm_a,
                            nm_g,
                            nm_b,
                            tolerance,
                            max_iterations,
                            store_trace,
                            show_trace)
@assert results.converged
@assert norm(results.minimum - [1.0, 1.0]) < 0.01

results = Optim.nelder_mead(rosenbrock, initial_p)
@assert results.converged
@assert norm(results.minimum - [1.0, 1.0]) < 0.01

initial_x = [0.0, 0.0]
results = Optim.nelder_mead(rosenbrock, initial_x)
@assert results.converged
@assert norm(results.minimum - [1.0, 1.0]) < 0.01

# Need to check that initial_p is an n * (n + 1) array.
