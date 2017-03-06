@testset "Momentum Gradient Descent" begin
    for use_autodiff in (false, true)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable && !(name in ("Polynomial", "Large Polynomial", "Himmelblau")) # it goes in a direction of ascent -> f_converged == true
                iterations = name == "Powell" ? 2000 : 1000
                res = Optim.optimize(prob.f, prob.initial_x, MomentumGradientDescent(),
                                     Optim.Options(autodiff = use_autodiff,
                                                         iterations = iterations))
                @test norm(Optim.minimizer(res) - prob.solutions, Inf) < 1e-2
            end
        end
    end
end
