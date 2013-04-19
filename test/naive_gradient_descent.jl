f(x::Vector) = (x[1] - 5.0)^2
function g!(x::Vector, storage::Vector)
    storage[1] = 2.0 * (x[1] - 5.0)
end

initial_x = [0.0]

store_trace, show_trace = false, false
results = Optim.naive_gradient_descent(f, g!, initial_x,
	                                   0.1, 10e-8, 1_000,
	                                   store_trace, show_trace)
@assert length(results.trace.states) == 0
@assert results.converged
@assert norm(results.minimum - [5.0]) < 0.01

results = Optim.naive_gradient_descent(f, g!, initial_x)
@assert length(results.trace.states) == 0
@assert results.converged
@assert norm(results.minimum - [5.0]) < 0.01

eta = 0.9

f(x::Vector) = (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)

function g!(x::Vector, storage::Vector)
    storage[1] = x[1]
    storage[2] = eta * x[2]
end

store_trace, show_trace = true, false
results = Optim.naive_gradient_descent(f, g!, [1.0, 1.0],
		                               0.1, 10e-8, 1_000,
	                                   store_trace, show_trace)
@assert length(results.trace.states) > 0
@assert results.converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
