using Optim

f(x) = x[1]^4
function g!(x, storage)
    storage[1] = 4 * x[1]^3
    return
end

d = DifferentiableFunction(f, g!)

initial_x = [1.0]

Optim.accelerated_gradient_descent(d,
	                               initial_x,
	                               show_trace = true,
	                               iterations = 10)
