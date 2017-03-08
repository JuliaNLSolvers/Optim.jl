@testset "L-BFGS" begin
    for (name, prob) in Optim.UnconstrainedProblems.examples
        if prob.isdifferentiable
            results = Optim.optimize(prob.f, prob.initial_x, LBFGS())
            if !(name in ("Rosenbrock", "Polynomial"))
                @test Optim.converged(results)
            end
            @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
        end
    end
end
