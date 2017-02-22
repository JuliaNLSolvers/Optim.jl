@testset "BFGS" begin
    for use_autodiff in (false, true)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            opt_allow_f_increases = name == "Polynomial" ? true : false
            if prob.isdifferentiable
                results = Optim.optimize(prob.f, prob.initial_x, BFGS(), Optim.Options(autodiff = use_autodiff, allow_f_increases = opt_allow_f_increases))
                if name != "Polynomial" && use_autodiff == false
                    @test Optim.converged(results)
                end
                @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
            end
        end
    end
end
