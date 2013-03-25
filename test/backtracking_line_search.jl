f(x::Vector) = (1.0 - x[1])^2
function g!(x::Vector, storage::Vector)
	storage[1] = -2.0 * (1.0 - x[1])
end

x = [0.0]
gradient = [0.0]
g!(x, gradient)
dx = -gradient

t, f_up, g_up = Optim.backtracking_line_search(f, g!, x, dx, 1e-6, 0.9, 0.9, 1_000)
@assert f(x + t * dx) < f(x) + alpha * t * -dot(gradient, dx)

t, f_up, g_up = Optim.backtracking_line_search(f, g!, x, dx)
@assert f(x + t * dx) < f(x) + alpha * t * -dot(gradient, dx)
