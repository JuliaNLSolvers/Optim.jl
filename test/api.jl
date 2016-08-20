# Test multivariate optimization
let
    rosenbrock = Optim.UnconstrainedProblems.examples["Rosenbrock"]
    f = rosenbrock.f
    g! = rosenbrock.g!
    h! = rosenbrock.h!
    initial_x = rosenbrock.initial_x

    d1 = DifferentiableFunction(f)
    d2 = DifferentiableFunction(f, g!)
    d3 = TwiceDifferentiableFunction(f, g!, h!)

    Optim.optimize(f, initial_x, BFGS())
    Optim.optimize(f, g!, initial_x, BFGS())
    Optim.optimize(f, g!, h!, initial_x, BFGS())
    Optim.optimize(d2, initial_x, BFGS())
    Optim.optimize(d3, initial_x, BFGS())

    Optim.optimize(f, initial_x, BFGS(), OptimizationOptions())
    Optim.optimize(f, g!, initial_x, BFGS(), OptimizationOptions())
    Optim.optimize(f, g!, h!, initial_x, BFGS(), OptimizationOptions())
    Optim.optimize(d2, initial_x, BFGS(), OptimizationOptions())
    Optim.optimize(d3, initial_x, BFGS(), OptimizationOptions())

    Optim.optimize(d1, initial_x, method = BFGS())
    Optim.optimize(d2, initial_x, method = BFGS())

    Optim.optimize(d1, initial_x, method = GradientDescent())
    Optim.optimize(d2, initial_x, method = GradientDescent())

    Optim.optimize(d1, initial_x, method = LBFGS())
    Optim.optimize(d2, initial_x, method = LBFGS())

    Optim.optimize(f, initial_x, method = NelderMead())

    Optim.optimize(d3, initial_x, method = Newton())

   Optim.optimize(f, initial_x, method = SimulatedAnnealing())

    optimize(f, initial_x, method = BFGS())
    optimize(f, initial_x, BFGS())
    optimize(f, g!, initial_x, method = BFGS())
    optimize(f, g!, h!, initial_x, method = BFGS())

    optimize(f, initial_x, method = GradientDescent())
    optimize(f, initial_x, GradientDescent())
    optimize(f, g!, initial_x, method = GradientDescent())
    optimize(f, g!, h!, initial_x, method = GradientDescent())

    optimize(f, initial_x, method = LBFGS())
    optimize(f, initial_x, LBFGS())
    optimize(f, g!, initial_x, method = LBFGS())
    optimize(f, g!, h!, initial_x, method = LBFGS())

    optimize(f, initial_x, method = NelderMead())
    optimize(f, initial_x, NelderMead())
    optimize(f, g!, initial_x, method = NelderMead())
    optimize(f, g!, h!, initial_x, method = NelderMead())

    optimize(f, g!, h!, initial_x, method = Newton())

    optimize(f, initial_x, method = SimulatedAnnealing())
    optimize(f, initial_x, SimulatedAnnealing())
    optimize(f, g!, initial_x, method = SimulatedAnnealing())
    optimize(f, g!, h!, initial_x, method = SimulatedAnnealing())

    res = optimize(f, g!, h!,
    	           initial_x,
    	           method = BFGS(),
    	           g_tol = 1e-12,
    	           iterations = 10,
    	           store_trace = true,
    	           show_trace = false)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           method = GradientDescent(),
    	           g_tol = 1e-12,
    	           iterations = 10,
    	           store_trace = true,
    	           show_trace = false)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           method = LBFGS(),
    	           g_tol = 1e-12,
    	           iterations = 10,
    	           store_trace = true,
    	           show_trace = false)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           method = NelderMead(),
    	           f_tol = 1e-12,
    	           iterations = 10,
    	           store_trace = true,
    	           show_trace = false)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           method = Newton(),
    	           g_tol = 1e-12,
    	           iterations = 10,
    	           store_trace = true,
    	           show_trace = false)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           method = SimulatedAnnealing(),
    	           iterations = 10,
    	           store_trace = true,
    	           show_trace = false)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           method = BFGS(),
    	           g_tol = 1e-12,
    	           iterations = 10,
    	           store_trace = true,
    	           show_trace = false)
   res_ext = optimize(f, g!, h!,
                      initial_x,
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

   res_extended = Optim.optimize(f, g!, initial_x, method=BFGS(), store_trace = true, extended_trace = true)
   @test haskey(Optim.trace(res_extended)[1].metadata,"~inv(H)")
   @test haskey(Optim.trace(res_extended)[1].metadata,"g(x)")
   @test haskey(Optim.trace(res_extended)[1].metadata,"x")

   res_extended_nm = Optim.optimize(f, g!, initial_x, method=NelderMead(), store_trace = true, extended_trace = true)
   @test haskey(Optim.trace(res_extended_nm)[1].metadata,"centroid")
   @test haskey(Optim.trace(res_extended_nm)[1].metadata,"step_type")
end

# Test univariate API
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
