@testset "Accelerated Gradient Descent" begin
    f(x) = x[1]^4
    function g!(x, storage)
        storage[1] = 4 * x[1]^3
        return
    end

    initial_x = [1.0]
    options = Optim.Options(show_trace = true, allow_f_increases=true)
    results = Optim.optimize(f, g!, initial_x, AcceleratedGradientDescent(), options)
    @test norm(Optim.minimum(results)) < 1e-6

    for (name, prob) in Optim.UnconstrainedProblems.examples
        if prob.isdifferentiable
            if !(name in ("Large Polynomial", "Parabola"))
                results = Optim.optimize(prob.f, prob.g!, prob.initial_x, AcceleratedGradientDescent(), Optim.Options(allow_f_increases=true))
                if !(name in ("Rosenbrock", "Polynomial", "Powell"))
                    @test Optim.converged(results)
                end
                @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
            end
        end
    end
end
