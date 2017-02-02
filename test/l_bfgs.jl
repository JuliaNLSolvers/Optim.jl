@testset "L-BFGS" begin
    for use_autodiff in (:finite, :forward, :reverse)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable
                f_prob = prob.f
                res = Optim.optimize(f_prob, prob.initial_x, LBFGS(), Optim.Options(autodiff = use_autodiff))
                @test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
            end
        end
    end
end
