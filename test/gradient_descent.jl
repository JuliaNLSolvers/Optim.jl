@testset "Gradient Descent" begin
    for use_autodiff in (false, true)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            opt_allow_f_increases = name == "Hosaki" ? true : false
            if prob.isdifferentiable
                iterations = if name == "Rosenbrock"
                        10000 # Zig-zagging
                    elseif name == "Powell"
                        80000 # Zig-zagging as the problem is (intentionally) ill-conditioned
                    else
                        1000
                end
                results = Optim.optimize(prob.f, prob.initial_x, GradientDescent(),
                                     Optim.Options(autodiff = use_autodiff,
                                                   iterations = iterations,
                                                   allow_f_increases = opt_allow_f_increases))
                if !(name in ("Rosenbrock", "Polynomial", "Powell"))
                    @test Optim.converged(results)
                end
                @test norm(Optim.minimizer(results) - prob.solutions, Inf) < 1e-2
            end
        end
    end

    function f_gd_1(x)
      (x[1] - 5.0)^2
    end

    function g_gd_1(x, storage)
      storage[1] = 2.0 * (x[1] - 5.0)
    end

    initial_x = [0.0]

    d = OnceDifferentiable(f_gd_1, g_gd_1)

    results = Optim.optimize(d, initial_x, GradientDescent())
    @test_throws ErrorException Optim.x_trace(results)
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [5.0]) < 0.01

    eta = 0.9

    function f_gd_2(x)
      (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    function g_gd_2(x, storage)
      storage[1] = x[1]
      storage[2] = eta * x[2]
    end

    d = OnceDifferentiable(f_gd_2, g_gd_2)

    results = Optim.optimize(d, [1.0, 1.0], GradientDescent())
    @test_throws ErrorException Optim.x_trace(results)
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01
end
