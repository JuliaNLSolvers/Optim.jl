f(x::Vector) = (1.0 - x[1])^2
function g!(x::Vector, storage::Vector)
	storage[1] = -2.0 * (1.0 - x[1])
end

x = [0.0]
gradient = [0.0]
ls_x = [0.0]
ls_gradient = [0.0]
g!(x, gradient)
dx = -gradient
d = DifferentiableFunction(f, g!)

t, f_up, g_up = Optim.backtracking_line_search!(d, x, dx, ls_x, ls_gradient)
@assert f(x + t * dx) < f(x) + 0.9 * t * -dot(gradient, dx)
