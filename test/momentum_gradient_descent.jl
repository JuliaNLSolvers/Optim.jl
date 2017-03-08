@testset "Momentum Gradient Descent" begin
    for (name, prob) in Optim.UnconstrainedProblems.examples
        iterations = name == "Powell" ? 2000 : 1000
        options = Optim.Options(iterations = iterations)
        if prob.isdifferentiable && !(name in ("Polynomial", "Large Polynomial", "Himmelblau")) # it goes in a direction of ascent -> f_converged == true
            res = Optim.optimize(prob.f, prob.initial_x, MomentumGradientDescent(), options)
            @test norm(Optim.minimizer(res) - prob.solutions, Inf) < 1e-2
            res = Optim.optimize(prob.f, prob.g!, prob.initial_x, MomentumGradientDescent(), options)
            @test norm(Optim.minimizer(res) - prob.solutions, Inf) < 1e-2
        end
    end
end
