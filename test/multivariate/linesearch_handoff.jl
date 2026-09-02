@testset "Line search hand-off" begin
    # `update_state!` must propose the point the line search vetted, so that `update_fgh!`
    # evaluates where the value is already cached and `accept_step!` checks the finiteness
    # of the point it commits.
    prob = MVP.UnconstrainedProblems.examples["Powell"]
    x0 = prob.initial_x
    options = Optim.Options()

    @testset "$(summary(method))" for method in
        (GradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), Newton())

        d = if method isa Optim.SecondOrderOptimizer
            TwiceDifferentiable(MVP.objective(prob), MVP.gradient(prob), MVP.hessian(prob), x0)
        else
            OnceDifferentiable(MVP.objective(prob), MVP.gradient(prob), x0)
        end
        state = Optim.initial_state(method, options, d, copy(x0))

        for _ = 1:5
            Optim.update_state!(d, state, method) && break
            f_calls = NLSolversBase.f_calls(d)
            Optim.update_fgh!(d, state, method)
            @test NLSolversBase.f_calls(d) == f_calls
            Optim.accept_step!(d, state, method, options) || break
        end
    end
end
