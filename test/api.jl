function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient!(x::Vector, storage::Vector)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

function rosenbrock_hessian!(x::Vector, storage::Matrix)
    storage[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    storage[1, 2] = -400.0 * x[1]
    storage[2, 1] = -400.0 * x[1]
    storage[2, 2] = 200.0
end

f3 = rosenbrock
g3! = rosenbrock_gradient!
h3! = rosenbrock_hessian!

d1 = DifferentiableFunction(rosenbrock)
d2 = DifferentiableFunction(rosenbrock,
	                        rosenbrock_gradient!)
d3 = TwiceDifferentiableFunction(rosenbrock,
	                             rosenbrock_gradient!,
	                             rosenbrock_hessian!)

Optim.bfgs(d1, [0.0, 0.0])
Optim.bfgs(d2, [0.0, 0.0])

Optim.gradient_descent(d1, [0.0, 0.0])
Optim.gradient_descent(d2, [0.0, 0.0])

Optim.l_bfgs(d1, [0.0, 0.0])
Optim.l_bfgs(d2, [0.0, 0.0])

Optim.nelder_mead(rosenbrock, [0.0, 0.0])

Optim.newton(d3, [0.0, 0.0])

Optim.simulated_annealing(rosenbrock, [0.0, 0.0])

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = :bfgs)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = :bfgs)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = :bfgs)

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = :gradient_descent)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = :gradient_descent)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = :gradient_descent)

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = :l_bfgs)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = :l_bfgs)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = :l_bfgs)

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = :nelder_mead)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = :nelder_mead)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = :nelder_mead)

optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = :newton)

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = :simulated_annealing)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = :simulated_annealing)
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = :simulated_annealing)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = :bfgs,
	           grtol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = :gradient_descent,
	           grtol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = :l_bfgs,
	           grtol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = :nelder_mead,
	           ftol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = :newton,
	           grtol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = :simulated_annealing,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)
