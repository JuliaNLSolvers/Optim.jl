# `dphi_0` has to be the slope the line search samples itself. `fjvp` here is the gradient
# version off by one ulp, so the two routes are distinguishable without either being wrong.
@testset "dphi_0 comes from the objective's JVP" begin
    fdf(F, G, x) = (G === nothing || (G .= 2 .* x); sum(abs2, x))
    fjvp(F, JVP, x, v) = (sum(abs2, x), nextfloat(dot(2 .* x, v)))

    x0 = [1.0, -2.0]
    d = OnceDifferentiable(NLSolversBase.InplaceObjective(; fdf, fjvp), x0)

    calls = Ref(0)
    linesearch = function (df, x, s, α, x_new, phi_0, dphi_0)
        calls[] += 1
        @test dphi_0 == last(LineSearches.make_ϕdϕ(df, x_new, x, s)(zero(α)))
        @test dphi_0 == NLSolversBase.jvp!(df, x, s)
        @test dphi_0 != dot(NLSolversBase.gradient!(df, x), s)
        x_new .= x .+ α .* s
        return α, phi_0
    end

    method = BFGS(; linesearch)
    state = Optim.initial_state(method, Optim.Options(), d, copy(x0))
    Optim.update_state!(d, state, method)
    @test calls[] == 1
end

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
