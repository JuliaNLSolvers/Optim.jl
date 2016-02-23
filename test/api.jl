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

Optim.optimize(f3, [0.0, 0.0], BFGS())
Optim.optimize(f3, g3!, [0.0, 0.0], BFGS())
Optim.optimize(f3, g3!, h3!, [0.0, 0.0], BFGS())
Optim.optimize(d2, [0.0, 0.0], BFGS())
Optim.optimize(d3, [0.0, 0.0], BFGS())

Optim.optimize(f3, [0.0, 0.0], BFGS(), OptimizationOptions())
Optim.optimize(f3, g3!, [0.0, 0.0], BFGS(), OptimizationOptions())
Optim.optimize(f3, g3!, h3!, [0.0, 0.0], BFGS(), OptimizationOptions())
Optim.optimize(d2, [0.0, 0.0], BFGS(), OptimizationOptions())
Optim.optimize(d3, [0.0, 0.0], BFGS(), OptimizationOptions())

Optim.optimize(d1, [0.0, 0.0], method = BFGS())
Optim.optimize(d2, [0.0, 0.0], method = BFGS())

Optim.optimize(d1, [0.0, 0.0], method = GradientDescent())
Optim.optimize(d2, [0.0, 0.0], method = GradientDescent())

Optim.optimize(d1, [0.0, 0.0], method = LBFGS())
Optim.optimize(d2, [0.0, 0.0], method = LBFGS())

Optim.optimize(rosenbrock, [0.0, 0.0], method = NelderMead())

Optim.optimize(d3, [0.0, 0.0], method = Newton())

Optim.optimize(rosenbrock, [0.0, 0.0], method = SimulatedAnnealing())

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = BFGS())
optimize(rosenbrock,
	     [0.0, 0.0],
	     BFGS())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = BFGS())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = BFGS())

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = GradientDescent())
optimize(rosenbrock,
	     [0.0, 0.0],
	     GradientDescent())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = GradientDescent())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = GradientDescent())

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = LBFGS())
optimize(rosenbrock,
	     [0.0, 0.0],
	     LBFGS())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = LBFGS())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = LBFGS())

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = NelderMead())
optimize(rosenbrock,
	     [0.0, 0.0],
	     NelderMead())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = NelderMead())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = NelderMead())

optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = Newton())

optimize(rosenbrock,
	     [0.0, 0.0],
	     method = SimulatedAnnealing())
optimize(rosenbrock,
	     [0.0, 0.0],
	     SimulatedAnnealing())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     [0.0, 0.0],
	     method = SimulatedAnnealing())
optimize(rosenbrock,
	     rosenbrock_gradient!,
	     rosenbrock_hessian!,
	     [0.0, 0.0],
	     method = SimulatedAnnealing())

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = BFGS(),
	           grtol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = GradientDescent(),
	           grtol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = LBFGS(),
	           grtol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = NelderMead(),
	           ftol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = Newton(),
	           grtol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = SimulatedAnnealing(),
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)
