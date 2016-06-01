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
	           g_tol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = GradientDescent(),
	           g_tol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = LBFGS(),
	           g_tol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = NelderMead(),
	           f_tol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = Newton(),
	           g_tol = 1e-12,
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

res = optimize(f3, g3!, h3!,
	           [0.0, 0.0],
	           method = SimulatedAnnealing(),
	           iterations = 10,
	           store_trace = true,
	           show_trace = false)

let
    res = optimize(f3, g3!, h3!,
    	           [0.0, 0.0],
    	           method = BFGS(),
    	           g_tol = 1e-12,
    	           iterations = 10,
    	           store_trace = true,
    	           show_trace = false)
   res_ext = optimize(f3, g3!, h3!,
                      [0.0, 0.0],
                      method = BFGS(),
                      g_tol = 1e-12,
                      iterations = 10,
                      store_trace = true,
                      show_trace = false,
                      extended_trace = true)
   @test Optim.method(res) == "BFGS"
   @test Optim.minimum(res) ≈ 0.055119582904897345
   @test Optim.minimizer(res) ≈ [0.7731690866149542; 0.5917345966396391]
   @test Optim.iterations(res) == 10
   @test Optim.f_calls(res) == 48
   @test Optim.g_calls(res) == 48
   @test Optim.converged(res) == false
   @test Optim.x_converged(res) == false
   @test Optim.f_converged(res) == false
   @test Optim.g_converged(res) == false
   @test Optim.x_tol(res) == 1e-32
   @test Optim.f_tol(res) == 1e-32
   @test Optim.g_tol(res) == 1e-12
   @test Optim.iteration_limit_reached(res) == true
   @test Optim.initial_state(res) == [0.0; 0.0]
   @test haskey(Optim.trace(res_ext)[1].metadata,"x")
   # just testing if it runs
   Optim.trace(res)
   Optim.f_trace(res)
   Optim.g_norm_trace(res)
   @test_throws ErrorException Optim.x_trace(res)
   @test_throws ErrorException Optim.x_lower_trace(res)
   @test_throws ErrorException Optim.x_upper_trace(res)
   @test_throws ErrorException Optim.lower_bound(res)
   @test_throws ErrorException Optim.upper_bound(res)
   @test_throws ErrorException Optim.rel_tol(res)
   @test_throws ErrorException Optim.abs_tol(res)

   res_extended = Optim.optimize(f3, g3!, [0.0, 0.0], method=BFGS(), store_trace = true, extended_trace = true)
   @test haskey(Optim.trace(res_extended)[1].metadata,"~inv(H)")
   @test haskey(Optim.trace(res_extended)[1].metadata,"g(x)")
   @test haskey(Optim.trace(res_extended)[1].metadata,"x")

   res_extended_nm = Optim.optimize(f3, g3!, [0.0, 0.0], method=NelderMead(), store_trace = true, extended_trace = true)
   @test haskey(Optim.trace(res_extended_nm)[1].metadata,"centroid")
   @test haskey(Optim.trace(res_extended_nm)[1].metadata,"step_type")
end

let
    f(x) = 2x^2+3x+1
    res = optimize(f, -2.0, 1.0, method = GoldenSection())
    @test Optim.method(res) == "Golden Section Search"
    @test Optim.minimum(res) ≈ -0.125
    @test Optim.minimizer(res) ≈ -0.749999994377939
    @test Optim.iterations(res) == 38
    @test Optim.iteration_limit_reached(res) == false
    @test_throws ErrorException Optim.trace(res)
    @test_throws ErrorException Optim.x_trace(res)
    @test_throws ErrorException Optim.x_lower_trace(res)
    @test_throws ErrorException Optim.x_upper_trace(res)
    @test_throws ErrorException Optim.f_trace(res)
    @test Optim.lower_bound(res) == -2.0
    @test Optim.upper_bound(res) == 1.0
    @test Optim.rel_tol(res) ≈ 1.4901161193847656e-8
    @test Optim.abs_tol(res) ≈ 2.220446049250313e-16
    @test_throws ErrorException Optim.initial_state(res)
    @test_throws ErrorException Optim.g_norm_trace(res)
    @test_throws ErrorException Optim.g_calls(res)
    @test_throws ErrorException Optim.x_converged(res)
    @test_throws ErrorException Optim.f_converged(res)
    @test_throws ErrorException Optim.g_converged(res)
    @test_throws ErrorException Optim.x_tol(res)
    @test_throws ErrorException Optim.f_tol(res)
    @test_throws ErrorException Optim.g_tol(res)
    res = optimize(f, -2.0, 1.0, method = GoldenSection(), store_trace = true, extended_trace = true)

    # Right now, these just "test" if they run
    Optim.x_trace(res)
    Optim.x_lower_trace(res)
    Optim.x_upper_trace(res)
end
