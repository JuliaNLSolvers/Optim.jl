@testset "L-BFGS" begin
    for use_autodiff in (false, true)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable
                results = Optim.optimize(prob.f, prob.initial_x, LBFGS(), Optim.Options(autodiff = use_autodiff))
                if !(name in ("Rosenbrock", "Polynomial"))
                    @test Optim.converged(results)
                end
                @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
            end
        end
    end
end
