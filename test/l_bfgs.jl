@testset "L-BFGS" begin
    for use_autodiff in (false, true)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable
                res = Optim.optimize(prob.f, prob.initial_x, LBFGS(), Optim.Options(autodiff = use_autodiff))
                @test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
            end
        end
    end
end
