let
    for use_autodiff in (false, true)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable
                f_prob = prob.f
                res = Optim.optimize(f_prob, prob.initial_x, BFGS(), OptimizationOptions(autodiff = use_autodiff))
                @assert norm(res.minimum - prob.solutions) < 1e-2
            end
        end
    end
end
