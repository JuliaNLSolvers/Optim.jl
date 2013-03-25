function f(x::Vector)
    (x[1] - 5.0)^4
end

function g!(x::Vector, storage::Vector)
    storage[1] = 4.0 * (x[1] - 5.0)^3
end

function h!(x::Vector, storage::Matrix)
    storage[1, 1] = 12.0 * (x[1] - 5.0)^2
end

store_trace, show_trace = false, false
results = Optim.newton(f,
	                   g!,
	                   h!,
	                   [0.0],
	                   10e-16,
	                   1_000,
	                   store_trace,
	                   show_trace)
@assert length(results.trace.states) == 0
@assert results.converged
@assert norm(results.minimum - [5.0]) < 0.01

results = Optim.newton(f, g!, h!, [0.0])
@assert length(results.trace.states) == 0
@assert results.converged
@assert norm(results.minimum - [5.0]) < 0.01

eta = 0.9

function f(x::Vector)
  (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g!(x::Vector, storage::Vector)
  storage[1] = x[1]
  storage[2] = eta * x[2]
end

function h!(x::Vector, storage::Matrix)
  storage[1, 1] = 1.0
  storage[1, 2] = 0.0
  storage[2, 1] = 0.0
  storage[2, 2] = eta
end

store_trace, show_trace = true, false
results = Optim.newton(f, g!, h!, [127.0, 921.0], 10e-16, 1_000,
	                   store_trace, show_trace)
@assert length(results.trace.states) > 0
@assert results.converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

store_trace, show_trace = false, false
results = Optim.newton(f, g!, h!, [127.0, 921.0])
@assert length(results.trace.states) == 0
@assert results.converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01
