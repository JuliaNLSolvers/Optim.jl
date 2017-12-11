@testset "Gradient Descent" begin
    run_optim_tests(GradientDescent(),
                    f_increase_exceptions = ("Hosaki",),
                    convergence_exceptions = (("Polynomial", 1), ("Polynomial", 2), ("Rosenbrock", 1), ("Rosenbrock", 2)),
                    iteration_exceptions = (("Rosenbrock", 10000),),
                    skip = ("Powell",))

    function f_gd_1(x)
      (x[1] - 5.0)^2
    end

    function g_gd_1(storage, x)
      storage[1] = 2.0 * (x[1] - 5.0)
    end

    initial_x = [0.0]

    d = OnceDifferentiable(f_gd_1, g_gd_1, 0.0, initial_x)

    results = Optim.optimize(d, initial_x, GradientDescent())
    @test_throws ErrorException Optim.x_trace(results)
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [5.0]) < 0.01
    @test summary(results) == "Gradient Descent"

    eta = 0.9

    function f_gd_2(x)
      (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    function g_gd_2(storage, x)
      storage[1] = x[1]
      storage[2] = eta * x[2]
    end

    d = OnceDifferentiable(f_gd_2, g_gd_2, 0.0, [1.0, 1.0])

    results = Optim.optimize(d, [1.0, 1.0], GradientDescent())
    @test_throws ErrorException Optim.x_trace(results)
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01
end
