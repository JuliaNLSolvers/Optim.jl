let
    for (name, prob) in Optim.UnivariateProblems.examples
        results = optimize(prob.f, prob.bounds..., method = GoldenSection())

        @assert Optim.converged(results)
        @assert norm(Optim.minimizer(results) - prob.minimizers) < 1e-7
    end
end
