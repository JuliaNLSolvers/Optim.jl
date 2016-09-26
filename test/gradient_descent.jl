let
    for use_autodiff in (false, true)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable
                f_prob = prob.f
                iterations = if name == "Rosenbrock"
                        5000 # Zig-zagging
                    elseif name == "Powell"
                        80000 # Zig-zagging as the problem is (intentionally) ill-conditioned
                    else
                        1000
                end
                res = Optim.optimize(f_prob, prob.initial_x, GradientDescent(),
                                     OptimizationOptions(autodiff = use_autodiff,
                                                         iterations = iterations))
                @assert norm(Optim.minimizer(res) - prob.solutions, Inf) < 1e-2
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

    d = DifferentiableFunction(f_gd_1, g_gd_1)

    results = Optim.optimize(d, initial_x, method=GradientDescent())
    @test_throws ErrorException Optim.x_trace(results)
    @assert Optim.g_converged(results)
    @assert norm(Optim.minimizer(results) - [5.0]) < 0.01

    eta = 0.9

    function f_gd_2(x)
      (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    function g_gd_2(x, storage)
      storage[1] = x[1]
      storage[2] = eta * x[2]
    end

    d = DifferentiableFunction(f_gd_2, g_gd_2)

    results = Optim.optimize(d, [1.0, 1.0], method=GradientDescent())
    @test_throws ErrorException Optim.x_trace(results)
    @assert Optim.g_converged(results)
    @assert norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01
end
