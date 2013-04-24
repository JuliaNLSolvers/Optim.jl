f(x) = (x[1] - 5.0)^2 + (x[2] - 11.0)^2
function g!(x, storage)
	storage[1] = 2.0 * (x[1] - 5.0)
	storage[2] = 2.0 * (x[2] - 11.0)
end

x = [0.0, 0.0]
x_new = [0.0, 0.0]
gr_new = [0.0, 0.0]
g!(x, gr_new)
p = -gr_new

d = DifferentiableFunction(f, g!)

Optim.interpolating_line_search!(d, x, p, x_new, gr_new)

Optim.l_bfgs(d, [0.0, 0.0])
