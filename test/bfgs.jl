@testset "BFGS" begin
    for (name, prob) in Optim.UnconstrainedProblems.examples
        opt_allow_f_increases = name == "Polynomial" ? true : false
        options = Optim.Options(allow_f_increases = opt_allow_f_increases)
        if prob.isdifferentiable
            results = Optim.optimize(prob.f, prob.initial_x, BFGS(), options)
            if name != "Polynomial"
                @test Optim.converged(results)
            end
            @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
            results = Optim.optimize(prob.f, prob.g!, prob.initial_x, BFGS(), options)
            if name != "Polynomial"
                @test Optim.converged(results)
            end
            @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
        end
    end
end
