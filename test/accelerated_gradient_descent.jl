@testset "Accelerated Gradient Descent" begin
    f(x) = x[1]^4
    function g!(x, storage)
        storage[1] = 4 * x[1]^3
        return
    end

    initial_x = [1.0]
    options = Optim.Options(show_trace = true, allow_f_increases=true)
    res = Optim.optimize(f, g!, initial_x, AcceleratedGradientDescent(), options)
    @test norm(Optim.minimum(res)) < 1e-6

    for (name, prob) in Optim.UnconstrainedProblems.examples
        if prob.isdifferentiable
            if !(name in ["Large Polynomial", "Parabola"])
                res = Optim.optimize(prob.f, prob.g!, prob.initial_x, AcceleratedGradientDescent(), Optim.Options(allow_f_increases=true))
                @test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
            end
        end
    end
end
