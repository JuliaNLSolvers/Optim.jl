# Test multivariate optimization
@testset "Multivariate API" begin
    rosenbrock = Optim.UnconstrainedProblems.examples["Rosenbrock"]
    f = rosenbrock.f
    g! = rosenbrock.g!
    h! = rosenbrock.h!
    initial_x = rosenbrock.initial_x

    d1 = OnceDifferentiable(f, initial_x)
    d2 = OnceDifferentiable(f, g!, initial_x)
    d3 = TwiceDifferentiable(f, g!, h!, initial_x)

    Optim.optimize(f, initial_x, BFGS())
    Optim.optimize(f, g!, initial_x, BFGS())
    Optim.optimize(f, g!, h!, initial_x, BFGS())
    Optim.optimize(d2, initial_x, BFGS())
    Optim.optimize(d3, initial_x, BFGS())

    Optim.optimize(f, initial_x, BFGS(), Optim.Options())
    Optim.optimize(f, g!, initial_x, BFGS(), Optim.Options())
    Optim.optimize(f, g!, h!, initial_x, BFGS(), Optim.Options())
    Optim.optimize(d2, initial_x, BFGS(), Optim.Options())
    Optim.optimize(d3, initial_x, BFGS(), Optim.Options())

    Optim.optimize(d1, initial_x, BFGS())
    Optim.optimize(d2, initial_x, BFGS())

    Optim.optimize(d1, initial_x, GradientDescent())
    Optim.optimize(d2, initial_x, GradientDescent())

    Optim.optimize(d1, initial_x, LBFGS())
    Optim.optimize(d2, initial_x, LBFGS())

    Optim.optimize(f, initial_x, NelderMead())

    Optim.optimize(d3, initial_x, Newton())

    Optim.optimize(f, initial_x, SimulatedAnnealing())

    optimize(f, initial_x, BFGS())
    optimize(f, g!, initial_x, BFGS())
    optimize(f, g!, h!, initial_x, BFGS())

    optimize(f, initial_x, GradientDescent())
    optimize(f, g!, initial_x, GradientDescent())
    optimize(f, g!, h!, initial_x, GradientDescent())

    optimize(f, initial_x, LBFGS())
    optimize(f, g!, initial_x, LBFGS())
    optimize(f, g!, h!, initial_x, LBFGS())

    optimize(f, initial_x, NelderMead())
    optimize(f, g!, initial_x, NelderMead())
    optimize(f, g!, h!, initial_x, NelderMead())

    optimize(f, g!, h!, initial_x, Newton())

    optimize(f, initial_x, SimulatedAnnealing())
    optimize(f, g!, initial_x, SimulatedAnnealing())
    optimize(f, g!, h!, initial_x, SimulatedAnnealing())

    options = Optim.Options(g_tol = 1e-12, iterations = 10,
                                  store_trace = true, show_trace = false)
    res = optimize(f, g!, h!,
    	           initial_x,
    	           BFGS(),
    	           options)

    options_g = Optim.Options(g_tol = 1e-12, iterations = 10,
                                  store_trace = true, show_trace = false)
    options_f = Optim.Options(g_tol = 1e-12, iterations = 10,
                                  store_trace = true, show_trace = false)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           GradientDescent(),
    	           options_g)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           LBFGS(),
    	           options_g)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           NelderMead(),
                   options_f)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           Newton(),
                   options_g)
    options_sa = Optim.Options(iterations = 10, store_trace = true,
                                     show_trace = false)
    res = optimize(f, g!, h!,
    	           initial_x,
    	           SimulatedAnnealing(),
    	           options_sa)

    res = optimize(f, g!, h!,
    	           initial_x,
    	           BFGS(),
    	           options_g)
    options_ext = Optim.Options(g_tol = 1e-12, iterations = 10,
                                      store_trace = true, show_trace = false,
                                      extended_trace = true)
    res_ext = optimize(f, g!, h!,
                       initial_x,
                       BFGS(),
                       options_ext)

   @test summary(res) == "BFGS"
   @test isapprox(Optim.minimum(res), 0.0020622412076141045; rtol=1e-3)
   @test isapprox(Optim.minimizer(res), [0.9719007353489979,0.9410235857510793]; rtol=1e-3)
   #@test isapprox(Optim.minimum(res), 6.556300e-02; rtol=1e-3) # BackTracking
   #@test isapprox(Optim.minimizer(res), [0.7458993932627154,0.553210145233104]; rtol=1e-3) # BackTracking
   @test Optim.iterations(res) == 10
   @test Optim.f_calls(res) == 37
   @test Optim.g_calls(res) == 37
   #@test Optim.f_calls(res) == 17 # BackTracking
   #@test Optim.g_calls(res) == 11 # BackTracking
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
   options_extended = Optim.Options(store_trace = true, extended_trace = true)
   res_extended = Optim.optimize(f, g!, initial_x, BFGS(), options_extended)
   @test haskey(Optim.trace(res_extended)[1].metadata,"~inv(H)")
   @test haskey(Optim.trace(res_extended)[1].metadata,"g(x)")
   @test haskey(Optim.trace(res_extended)[1].metadata,"x")
   options_extended_nm = Optim.Options(store_trace = true, extended_trace = true)
   res_extended_nm = Optim.optimize(f, g!, initial_x, NelderMead(), options_extended_nm)
   @test haskey(Optim.trace(res_extended_nm)[1].metadata,"centroid")
   @test haskey(Optim.trace(res_extended_nm)[1].metadata,"step_type")
end

# Test univariate API
@testset "Univariate API" begin
    f(x) = 2x^2+3x+1
    res = optimize(f, -2.0, 1.0, GoldenSection())
    @test summary(res) == "Golden Section Search"
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
    options =
    res = optimize(f, -2.0, 1.0, GoldenSection(), store_trace = true, extended_trace = true)

    # Right now, these just "test" if they run
    Optim.x_trace(res)
    Optim.x_lower_trace(res)
    Optim.x_upper_trace(res)
end
