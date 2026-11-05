@testset "Newton decrement" begin
    # f(x) = xᵀAx/2 - bᵀx with A positive definite, so λ²/2 = f(x) - f* exactly
    n = 5
    A = [i == j ? 2.0 + i : 0.5^abs(i - j) for i = 1:n, j = 1:n]
    b = collect(1.0:n)
    fstar = -dot(b, A \ b) / 2
    fquad(x) = dot(x, A * x) / 2 - dot(b, x)
    gquad!(G, x) = (G .= A * x .- b; G)
    hquad!(H, x) = (H .= A; H)
    hvquad!(Hv, x, v) = (Hv .= A * v; Hv)
    fgquad!(F, G, x) = (G === nothing || gquad!(G, x); F === nothing ? nothing : fquad(x))
    x0 = fill(3.0, n)

    quad_objective(::Optim.FirstOrderOptimizer) = OnceDifferentiable(fquad, gquad!, x0)
    quad_objective(::NewtonTrustRegion) = TwiceDifferentiable(fquad, gquad!, hquad!, x0)
    quad_objective(::Optim.KrylovTrustRegion) =
        TwiceDifferentiable(NLSolversBase.only_fg_and_hvp!(fgquad!, hvquad!), x0)

    @testset "stops on the decrement: $(summary(method))" for method in
        (BFGS(), LBFGS(), NewtonTrustRegion(), Optim.KrylovTrustRegion(cg_tol = 1e-8))

        options = Optim.Options(g_abstol = NaN, newton_decrement_abstol = 1e-12)
        r = optimize(quad_objective(method), copy(x0), method, options)
        @test Optim.termination_code(r) == Optim.TerminationCode.NewtonDecrement
        @test Optim.converged(r)
        @test Optim.newton_decrement_converged(r)
        @test Optim.newton_decrement(r) <= 1e-12
        @test Optim.minimum(r) - fstar <= 1e-10
        @test occursin("λ²/2", sprint(show, r))
    end

    @testset "a zero gradient has a zero decrement: $(summary(method))" for method in
        (BFGS(), LBFGS(), NewtonTrustRegion(), Optim.KrylovTrustRegion())

        # The first step lands exactly on the minimizer, where gᵀs = 0
        prob = MVP.UnconstrainedProblems.examples["Parabola"]
        f, g!, h! = MVP.objective(prob), MVP.gradient(prob), MVP.hessian(prob)
        d = method isa Optim.FirstOrderOptimizer ? OnceDifferentiable(f, g!, prob.initial_x) :
            TwiceDifferentiable(f, g!, h!, prob.initial_x)
        options = Optim.Options(g_abstol = NaN, newton_decrement_abstol = 1e-12)
        r = optimize(d, copy(prob.initial_x), method, options)
        @test Optim.termination_code(r) == Optim.TerminationCode.NewtonDecrement
        @test Optim.minimum(r) == prob.minimum
    end

    @testset "the decrement is f(x) - f* on a quadratic" begin
        # The exact Hessian makes the decrement at x0 the gap itself, so a tolerance just
        # above it stops the run before the first step
        gap = fquad(x0) - fstar
        options = Optim.Options(newton_decrement_abstol = 1.01 * gap)
        r = optimize(fquad, gquad!, hquad!, copy(x0), NewtonTrustRegion(), options)
        @test Optim.iterations(r) == 0
        @test Optim.termination_code(r) == Optim.TerminationCode.NewtonDecrement
        @test Optim.newton_decrement(r) ≈ gap rtol = 1e-12
    end

    @testset "off by default: $(summary(method))" for method in
        (BFGS(), LBFGS(), NewtonTrustRegion(), Optim.KrylovTrustRegion())

        r = optimize(quad_objective(method), copy(x0), method, Optim.Options())
        @test isnan(Optim.newton_decrement(r))
        @test !Optim.newton_decrement_converged(r)
        @test Optim.termination_code(r) == Optim.TerminationCode.GradientNorm
        @test !occursin("λ²/2", sprint(show, r))
    end

    @testset "no curvature from the initial matrix: $(summary(method))" for method in
        (BFGS(), LBFGS())

        # With the identity in place of the inverse Hessian, -gᵀs/2 at x0 is 2e-6 while
        # f(x0) - f* = 2
        fs(x) = 1e-6 * sum(abs2, x .- 1e3)
        gs!(G, x) = (G .= 2e-6 .* (x .- 1e3); G)
        options = Optim.Options(g_abstol = NaN, newton_decrement_abstol = 1e-5)
        r = optimize(fs, gs!, zeros(2), method, options)
        @test Optim.iterations(r) >= 1
        @test Optim.minimum(r) <= 1e-4
    end

    @testset "no stop on negative curvature" begin
        # A double well started in its concave middle. A backtracking line search does not
        # enforce the curvature condition, so the first pair has dot(dx, dg) < 0.
        fw(x) = sum(xi -> xi^4 / 4 - xi^2 / 2, x)
        gw!(G, x) = (G .= x .^ 3 .- x; G)
        method = LBFGS(linesearch = LineSearches.BackTracking())
        options = Optim.Options(g_abstol = NaN, newton_decrement_abstol = 1e-3)
        r = optimize(fw, gw!, [0.1], method, options)
        @test abs(only(Optim.minimizer(r))) > 0.9
    end

    @testset "abstol and reltol" begin
        # λ²/2 = (2/3)Σxᵢ⁴ under the exact Hessian of Σxᵢ⁴, so the relative decrement of
        # f = Σxᵢ⁴ stays at 2/3 all the way to its minimum at 0
        f4(x) = sum(abs2 ∘ abs2, x)
        g4!(G, x) = (G .= 4 .* x .^ 3; G)
        h4!(H, x) = (H .= Diagonal(12 .* x .^ 2); H)
        x4 = [1.0, 2.0]

        r = optimize(f4, g4!, h4!, copy(x4), NewtonTrustRegion(),
            Optim.Options(newton_decrement_reltol = 1e-8))
        @test Optim.termination_code(r) == Optim.TerminationCode.GradientNorm
        @test Optim.newton_decrement(r) / Optim.minimum(r) ≈ 2 / 3 rtol = 1e-8

        r = optimize(f4, g4!, h4!, copy(x4), NewtonTrustRegion(),
            Optim.Options(g_abstol = NaN, newton_decrement_abstol = 1e-10))
        @test Optim.termination_code(r) == Optim.TerminationCode.NewtonDecrement
        @test Optim.minimum(r) <= 1.5e-10

        # Shifted away from zero, the relative test fires
        r = optimize(x -> f4(x) + 1, g4!, h4!, copy(x4), NewtonTrustRegion(),
            Optim.Options(g_abstol = NaN, newton_decrement_reltol = 1e-10))
        @test Optim.termination_code(r) == Optim.TerminationCode.NewtonDecrement
        @test Optim.newton_decrement(r) <= 1e-10 * Optim.minimum(r)
        @test occursin("λ²/2/|f(x)|", sprint(show, r))
    end

    @testset "a state passed back in: $(summary(method))" for method in (BFGS(), LBFGS())
        # Evaluating the decrement at the returned point must not disturb the state
        prob = MVP.UnconstrainedProblems.examples["Rosenbrock"]
        d = OnceDifferentiable(MVP.objective(prob), MVP.gradient(prob), prob.initial_x)
        options(k) = Optim.Options(g_abstol = NaN, newton_decrement_abstol = 1e-300, iterations = k)
        r10 = optimize(d, copy(prob.initial_x), method, options(10))
        state = Optim.initial_state(method, options(5), d, copy(prob.initial_x))
        optimize(d, copy(prob.initial_x), method, options(5), state)
        r55 = optimize(d, copy(prob.initial_x), method, options(5), state)
        @test Optim.minimizer(r55) == Optim.minimizer(r10)
    end

    @testset "Fminbox with a supported inner method" begin
        options = Optim.Options(newton_decrement_abstol = 1e-12)
        r = optimize(fquad, gquad!, fill(-10.0, n), fill(0.5, n), zeros(n), Fminbox(LBFGS()), options)
        rref = optimize(fquad, gquad!, fill(-10.0, n), fill(0.5, n), zeros(n), Fminbox(LBFGS()))
        @test Optim.converged(r)
        @test Optim.minimizer(r) ≈ Optim.minimizer(rref) atol = 1e-6
    end

    @testset "maximize" begin
        r = maximize(x -> -fquad(x), (G, x) -> (gquad!(G, x); G .= .-G), copy(x0), BFGS(),
            Optim.Options(g_abstol = NaN, newton_decrement_abstol = 1e-12))
        @test Optim.newton_decrement_converged(r)
        @test Optim.newton_decrement(r) <= 1e-12
    end

    @testset "unsupported methods throw" begin
        for options in (
            Optim.Options(newton_decrement_abstol = 1e-8),
            Optim.Options(newton_decrement_reltol = 1e-8),
        )
            for method in (GradientDescent(), ConjugateGradient(), AcceleratedGradientDescent(),
                NGMRES(), Newton())
                @test_throws ArgumentError optimize(fquad, gquad!, hquad!, copy(x0), method, options)
            end
            @test_throws ArgumentError optimize(fquad, copy(x0), NelderMead(), options)
            lb, ub = fill(-10.0, n), fill(10.0, n)
            @test_throws ArgumentError optimize(fquad, gquad!, lb, ub, zeros(n), LBFGSB(), options)
            @test_throws ArgumentError optimize(
                fquad, gquad!, lb, ub, zeros(n), Fminbox(GradientDescent()), options,
            )
            @test_throws ArgumentError optimize(fquad, lb, ub, zeros(n), SAMIN(), options)
            df = TwiceDifferentiable(fquad, gquad!, hquad!, zeros(n))
            dfc = TwiceDifferentiableConstraints(lb, ub)
            @test_throws ArgumentError optimize(df, dfc, zeros(n), IPNewton(), options)
        end
    end
end
