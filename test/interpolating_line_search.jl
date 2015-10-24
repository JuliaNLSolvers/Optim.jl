f5(x) = (x[1] - 5.0)^2 + (x[2] - 11.0)^2
function g5!(x, storage)
	storage[1] = 2.0 * (x[1] - 5.0)
	storage[2] = 2.0 * (x[2] - 11.0)
end
d = DifferentiableFunction(f5, g5!)

x = [0.0, 0.0]
x_new = [0.0, 0.0]
gr_new = [0.0, 0.0]
phi0 = d.fg!(x, gr_new)
p = -gr_new
dphi0 = dot(p, gr_new)

lsr = Optim.LineSearchResults(eltype(x))
push!(lsr, 0.0, phi0, dphi0)

alpha = 1.0
mayterminate = false
alpha, f_update, g_update = Optim.backtracking_linesearch!(d, x, p, x_new, gr_new, lsr, alpha, mayterminate)
alpha, f_update, g_update = Optim.interpolating_linesearch!(d, x, p, x_new, gr_new, lsr, alpha, mayterminate)
alpha, f_update, g_update = Optim.mt_linesearch!(d, x, p, x_new, gr_new, lsr, alpha, mayterminate)
alpha, f_update, g_update = Optim.hz_linesearch!(d, x, p, x_new, gr_new, lsr, alpha, mayterminate)

Optim.optimize(d, [0.0, 0.0], method=LBFGS())
