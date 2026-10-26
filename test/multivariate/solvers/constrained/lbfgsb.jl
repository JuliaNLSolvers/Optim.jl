# Reference Fortran L-BFGS-B 3.0 wrapper. Aliased so it does not collide with
# Optim's exported `LBFGSB` (the native solver under test).
import LBFGSB as RefLBFGSB

@testset "L-BFGS-B" begin
    feasible(x, l, u) = all(l .<= x .<= u)

    # Run the whole suite against both the efficient `LBFGSB` and the internal
    # reference `Optim.SimpleLBFGSBReference.SimpleLBFGSB`. Each variant is the
    # solver type itself, so `V(m = m)` constructs it; its line searches live in
    # the type's own module (`parentmodule(V).HZAW()` / `.MTLS()`).
    variants = (LBFGSB, Optim.SimpleLBFGSBReference.SimpleLBFGSB)

    # Reference for "agreement with Fminbox(LBFGS())", the same for both variants
    fminbox_rosen = optimize(
        x -> (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2,
        [-2.0, -2.0],
        [2.0, 2.0],
        [-1.2, 1.0],
        Fminbox(LBFGS()),
    )

    @testset "$(nameof(V))" for V in variants
        @testset "unconstrained problems with slack bounds" begin
            for (name, prob) in MVP.UnconstrainedProblems.examples
                xtrue = prob.solutions
                length(xtrue) > 10 && continue
                ("Large Polynomial" == name || "Trigonometric" == name) && continue
                f = MVP.objective(prob)
                l = min.(xtrue, prob.initial_x) .- 1
                u = max.(xtrue, prob.initial_x) .+ 1
                res = optimize(f, l, u, prob.initial_x, V(), Optim.Options(g_abstol = 1e-8))
                @test feasible(Optim.minimizer(res), l, u)
                @test abs(prob.minimum - Optim.minimum(res)) < 1e-6
            end
        end

        @testset "active bound at the solution" begin
            g(x) = (x[1] - 3.0)^2 + (x[2] + 4.0)^2
            l = [-2.0, -2.0]
            u = [2.0, 2.0]
            res = optimize(g, l, u, [0.0, 0.0], V())
            @test feasible(Optim.minimizer(res), l, u)
            @test Optim.minimizer(res) ≈ [2.0, -2.0] atol = 1e-6
        end

        @testset "agreement with Fminbox(LBFGS())" begin
            rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
            l = [-2.0, -2.0]
            u = [2.0, 2.0]
            x0 = [-1.2, 1.0]
            rb = optimize(rosen, l, u, x0, V())
            @test Optim.minimizer(rb) ≈ [1.0, 1.0] atol = 1e-4
            @test Optim.minimizer(rb) ≈ Optim.minimizer(fminbox_rosen) atol = 1e-3
        end

        @testset "line searches" begin
            rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
            l = [-2.0, -2.0]
            u = [2.0, 2.0]
            x0 = [-1.2, 1.0]
            for ls in (parentmodule(V).HZAW(), parentmodule(V).MTLS())
                res = optimize(rosen, l, u, x0, V(linesearch = ls))
                @test feasible(Optim.minimizer(res), l, u)
                @test Optim.minimizer(res) ≈ [1.0, 1.0] atol = 1e-4
            end
        end

        @testset "scalar bounds" begin
            f(x) = sum(abs2, x)
            x0 = [1.0, -2.0, 3.0]
            res = optimize(f, -5.0, 5.0, x0, V())
            @test Optim.converged(res)
            @test Optim.minimizer(res) ≈ zeros(3) atol = 1e-6
        end

        @testset "one-sided bounds and fixed variables" begin
            g(x) = (x[1] - 3.0)^2 + (x[2] + 4.0)^2
            res = optimize(g, [1.0, -Inf], [Inf, Inf], [3.0, 0.0], V())
            @test Optim.minimizer(res) ≈ [3.0, -4.0] atol = 1e-6

            res2 = optimize(g, [1.0, -10.0], [1.0, 10.0], [1.0, 0.0], V())
            @test Optim.minimizer(res2)[1] ≈ 1.0 atol = 1e-8
            @test Optim.minimizer(res2)[2] ≈ -4.0 atol = 1e-6
        end

        @testset "type genericity" begin
            f(x) = sum(abs2, x)
            for ls in (parentmodule(V).HZAW(), parentmodule(V).MTLS())
                res32 = optimize(f, fill(-5.0f0, 3), fill(5.0f0, 3), Float32[1, -2, 3], V(linesearch = ls))
                @test eltype(Optim.minimizer(res32)) == Float32
                @test typeof(Optim.minimum(res32)) == Float32
                @test Optim.minimum(res32) < 1.0f-8
            end
        end

        @testset "options: iteration cap and callback" begin
            rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
            l = [-2.0, -2.0]
            u = [2.0, 2.0]
            capped = optimize(rosen, l, u, [-1.2, 1.0], V(), Optim.Options(iterations = 2))
            @test Optim.iterations(capped) <= 2
            @test !Optim.converged(capped)

            calls = Ref(0)
            cb = state -> (calls[] += 1; state.iteration >= 3)
            optimize(rosen, l, u, [-1.2, 1.0], V(), Optim.Options(callback = cb))
            @test calls[] >= 1
        end

        @testset "errors on infeasible start / bad bounds" begin
            f(x) = sum(abs2, x)
            @test_throws ArgumentError optimize(f, [0.0, 0.0], [1.0, 1.0], [2.0, 0.5], V())
            @test_throws ArgumentError optimize(f, [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], V())
        end

        @testset "high-dimensional active set, clip=$clip" for clip in (true, false)
            n = 30
            c = collect(range(-3.0, 3.0; length = n))
            f(x) = sum((x .- c) .^ 2)
            l = fill(-1.0, n)
            u = fill(1.0, n)
            xstar = clamp.(c, l, u)
            res = optimize(f, l, u, zeros(n), V(clip_subspace = clip), Optim.Options(g_abstol = 1e-8))
            @test feasible(Optim.minimizer(res), l, u)
            @test Optim.minimizer(res) ≈ xstar atol = 1e-6
            @test any(Optim.minimizer(res) .≈ -1.0)
            @test any(Optim.minimizer(res) .≈ 1.0)
        end

        @testset "iterate stays strictly in the box at active bounds" begin
            # The committed step is projected onto [l, u], so the returned
            # minimizer must satisfy l .<= x .<= u with NO tolerance, and the
            # active coordinates must land exactly on a bound (not a ULP outside).
            g(x) = (x[1] - 3.0)^2 + (x[2] + 4.0)^2   # unconstrained min [3, -4]
            l2, u2 = [-2.0, -2.0], [2.0, 2.0]
            r1 = optimize(g, l2, u2, [0.0, 0.0], V())
            @test all(l2 .<= Optim.minimizer(r1) .<= u2)   # strict, no tolerance

            n = 20
            c = collect(range(-3.0, 3.0; length = n))     # ~2/3 of bounds active
            f(x) = sum((x .- c) .^ 2)
            lo, hi = fill(-1.0, n), fill(1.0, n)
            r2 = optimize(f, lo, hi, zeros(n), V(), Optim.Options(g_abstol = 1e-8))
            x2 = Optim.minimizer(r2)
            @test all(lo .<= x2 .<= hi)                    # strict, no tolerance
            # every coordinate is exactly on a bound or strictly interior
            @test all(@. (x2 == -1.0) | (x2 == 1.0) | (-1.0 < x2 < 1.0))
        end

        @testset "feasible start on the boundary" begin
            n = 10
            c = collect(range(-3.0, 3.0; length = n))
            f(x) = sum((x .- c) .^ 2)
            l = fill(-1.0, n)
            u = fill(1.0, n)
            res = optimize(f, l, u, fill(-1.0, n), V())
            @test feasible(Optim.minimizer(res), l, u)
            @test Optim.minimizer(res) ≈ clamp.(c, l, u) atol = 1e-6
        end

        @testset "memory length m=$m" for m in (2, 4, 20)
            rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
            res = optimize(rosen, [-2.0, -2.0], [2.0, 2.0], [-1.2, 1.0], V(m = m), Optim.Options(iterations = 2000, g_abstol = 1e-8))
            @test Optim.converged(res)
            @test Optim.minimizer(res) ≈ [1.0, 1.0] atol = 1e-4
        end

        @testset "memory length m=1 reaches the minimizer" begin
            # A single correction pair gives too poor a model to drive the projected
            # gradient below g_abstol on Rosenbrock for the efficient solver (the
            # Fortran reference behaves the same); the minimizer is still reached.
            rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
            res = optimize(rosen, [-2.0, -2.0], [2.0, 2.0], [-1.2, 1.0], V(m = 1))
            @test Optim.minimizer(res) ≈ [1.0, 1.0] atol = 1e-4
        end

        @testset "tracing" begin
            rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
            l = [-2.0, -2.0]
            u = [2.0, 2.0]

            ext = optimize(rosen, l, u, [-1.2, 1.0], V(),
                Optim.Options(store_trace = true, extended_trace = true))
            tr = Optim.trace(ext)
            @test length(tr) >= 1
            @test isfinite(tr[end].g_norm)
            @test haskey(tr[end].metadata, "x")
            @test haskey(tr[end].metadata, "g(x)")

            plain = optimize(rosen, l, u, [-1.2, 1.0], V(), Optim.Options(store_trace = true))
            @test !haskey(Optim.trace(plain)[end].metadata, "x")
        end

        @testset "termination by limits" begin
            rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
            l = [-2.0, -2.0]
            u = [2.0, 2.0]
            x0 = [-1.2, 1.0]

            r_f = optimize(rosen, l, u, x0, V(), Optim.Options(f_calls_limit = 5))
            @test !Optim.converged(r_f)
            @test r_f.termination_code == Optim.TerminationCode.ObjectiveCalls

            r_g = optimize(rosen, l, u, x0, V(), Optim.Options(g_calls_limit = 5))
            @test !Optim.converged(r_g)
            @test r_g.termination_code == Optim.TerminationCode.GradientCalls

            r_i = optimize(rosen, l, u, x0, V(), Optim.Options(iterations = 3))
            @test Optim.iterations(r_i) == 3
            @test r_i.termination_code == Optim.TerminationCode.Iterations
        end

        @testset "termination codes for x/f tolerances" begin
            # With g_abstol disabled, convergence is on the x- or f-change criterion,
            # and the reported termination_code must reflect that (regression: the
            # state passed to _termination_code must not subtype ZerothOrderState).
            rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
            g!(G, x) = (G[1] = -2.0 * (1 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2);
                G[2] = 200.0 * (x[2] - x[1]^2); G)
            l = [-2.0, -2.0]
            u = [2.0, 2.0]
            x0 = [-1.2, 1.0]

            r_f = optimize(rosen, g!, l, u, x0, V(), Optim.Options(g_abstol = 0.0, f_abstol = 1e-6))
            @test Optim.converged(r_f)
            @test r_f.termination_code == Optim.TerminationCode.SmallObjectiveChange

            r_x = optimize(rosen, g!, l, u, x0, V(), Optim.Options(g_abstol = 0.0, x_abstol = 1e-4))
            @test Optim.converged(r_x)
            @test r_x.termination_code == Optim.TerminationCode.SmallXChange
        end

        @testset "BigFloat precision" begin
            f(x) = sum(abs2, x)
            res = optimize(f, fill(big(-5.0), 3), fill(big(5.0), 3), big.([1.0, -2.0, 3.0]),
                V(), Optim.Options(g_abstol = big(1e-20)))
            @test eltype(Optim.minimizer(res)) == BigFloat
            @test Optim.minimum(res) < big(1e-18)
        end

        @testset "active set under $T" for T in (Float32, BigFloat)
            # Heavily active box (~2/3 of the bounds active at the solution) under a
            # non-Float64 eltype, exercising the Cauchy point + subspace solve
            # (Cholesky formk!/bmv! for the efficient port, dense inv for the
            # reference) end-to-end in T. min Σ(xᵢ - cᵢ)² over [-1, 1]ⁿ has the
            # analytic minimizer clamp.(c, l, u).
            n = 12
            c = collect(range(T(-3), T(3); length = n))
            f(x) = sum((x .- c) .^ 2)
            l = fill(T(-1), n)
            u = fill(T(1), n)
            xstar = clamp.(c, l, u)
            gtol, atol = T === BigFloat ? (T(1e-20), T(1e-12)) : (1.0f-5, 1.0f-4)
            res = optimize(f, l, u, zeros(T, n), V(), Optim.Options(g_abstol = gtol))
            @test eltype(Optim.minimizer(res)) == T
            @test feasible(Optim.minimizer(res), l, u)
            @test Optim.minimizer(res) ≈ xstar atol = atol
            @test any(Optim.minimizer(res) .≈ T(-1))   # lower bounds active
            @test any(Optim.minimizer(res) .≈ T(1))    # upper bounds active
        end

        @testset "cross-check vs reference LBFGSB.jl (Fortran)" begin
            refkw = (m = 10, factr = 1e1, pgtol = 1e-9, maxiter = 10_000, maxfun = 10_000)

            csep = collect(range(-3.0, 3.0; length = 25))
            problems = [
                (
                    name = "rosenbrock (interior minimum)",
                    fval = x -> (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2,
                    grad = x -> [
                        -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2),
                        200.0 * (x[2] - x[1]^2),
                    ],
                    l = [-2.0, -2.0],
                    u = [2.0, 2.0],
                    x0 = [-1.2, 1.0],
                ),
                (
                    name = "shifted quadratic (active corner)",
                    fval = x -> (x[1] - 3.0)^2 + (x[2] + 4.0)^2,
                    grad = x -> [2.0 * (x[1] - 3.0), 2.0 * (x[2] + 4.0)],
                    l = [-2.0, -2.0],
                    u = [2.0, 2.0],
                    x0 = [0.0, 0.0],
                ),
                (
                    name = "separable quadratic (many active)",
                    fval = x -> sum((x .- csep) .^ 2),
                    grad = x -> 2.0 .* (x .- csep),
                    l = fill(-1.0, length(csep)),
                    u = fill(1.0, length(csep)),
                    x0 = zeros(length(csep)),
                ),
            ]

            for p in problems
                @testset "$(p.name)" begin
                    fref, xref = RefLBFGSB.lbfgsb(
                        x -> (p.fval(x), p.grad(x)),
                        p.x0;
                        lb = p.l,
                        ub = p.u,
                        refkw...,
                    )

                    g! = (G, x) -> copyto!(G, p.grad(x))
                    res = optimize(p.fval, g!, p.l, p.u, p.x0, V(), Optim.Options(g_abstol = 1e-9))

                    @test feasible(Optim.minimizer(res), p.l, p.u)
                    @test Optim.minimizer(res) ≈ xref atol = 1e-4
                    @test Optim.minimum(res) ≈ fref atol = 1e-8 rtol = 1e-6
                end
            end
        end
    end

    @testset "subspace backtracking measures from the Cauchy point" begin
        # Box-constrained Rosenbrock instance where the clipped subspace step
        # hits a bound and the direction to the clamped point is uphill
        # (dd_p > 0 in subspace_optimize!). The backtracking safeguard must
        # measure the step to the bounds from the unclamped Cauchy point;
        # measuring from the already-clamped values collapses the step onto the
        # flagged point, and the solver then recovers only by discarding its
        # full history (65 objective evaluations on this instance instead of 44).
        rosen2(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        function rosen2_g!(G, x)
            G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
            G[2] = 200.0 * (x[2] - x[1]^2)
            G
        end
        l = [-0.6246061824346689, -1.3848789693740542]
        u = [1.2952187059881008, 0.42144488961717363]
        x0 = [-0.6246061824346689, 0.42144488961717363]
        res = optimize(rosen2, rosen2_g!, l, u, x0, LBFGSB(m = 5))
        @test Optim.converged(res)
        @test feasible(Optim.minimizer(res), l, u)
        @test Optim.minimum(res) ≈ 0.12234569775490675 atol = 1e-8
        @test Optim.f_calls(res) <= 55
    end

    @testset "with every variable free the step is the two-loop L-BFGS direction" begin
        # Five updates into a memory of three, so the buffer is full and rotated; with
        # all variables free and the subspace step started at x, the step must be
        # -H g for the L-BFGS H built from the last three pairs with H0 = I/θ.
        n, m = 7, 3
        B = Optim.CompactLBFGS{Float64}(n, m)
        pairs = Tuple{Vector{Float64},Vector{Float64}}[]
        updatd = false
        for k in 1:5
            s = [sin(k * i) + 0.1 * k for i in 1:n]
            y = s .* (1 .+ (1:n) ./ n) .+ 0.01 .* cos.(k .* (1:n))
            push!(pairs, (s, y))
            yts = Optim.update_compact!(B, zeros(n), s, zeros(n), y)
            @test yts > 0
            fill!(B.iwhere, 0)
            nenter, ileave, wrk = Optim.freev!(B, k, true)
            @test Optim.formk!(B, nenter, ileave, true) == 0
        end
        @test B.k == m && B.head != 1
        g = [cos(2i) for i in 1:n]
        x = zeros(n)
        copyto!(B.xc, x)
        fill!(B.c_ws, 0.0)
        Optim.subspace_optimize!(B, x, g, fill(-Inf, n), fill(Inf, n))
        q = copy(g)
        recent = pairs[end-m+1:end]
        α = zeros(m)
        for j in m:-1:1
            s, y = recent[j]
            α[j] = dot(s, q) / dot(y, s)
            q .-= α[j] .* y
        end
        q ./= B.θ
        for j in 1:m
            s, y = recent[j]
            β = dot(y, q) / dot(y, s)
            q .+= (α[j] - β) .* s
        end
        @test B.xc - x ≈ -q rtol = 1e-10
    end

    @testset "an indefinite T refreshes the memory" begin
        # T = θ SᵀS + L D⁻¹ Lᵀ is positive definite in exact arithmetic but can lose that
        # in floating point when the stored steps are nearly dependent, as on Powell's
        # badly scaled function. The Fortran then refreshes the memory and restarts from
        # B = I. Corrupt the stored SᵀS so that T is indefinite after the next update.
        B = Optim.CompactLBFGS{Float64}(2, 5)
        Optim.update_compact!(B, [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0])
        @test B.k == 1
        B.SᵀS[1, 1] = -1.0
        yts = Optim.update_compact!(B, [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0])
        @test iszero(yts)
        @test B.k == 0
        @test B.θ == 1
        @test B.iupdat == 0
    end

    @testset "each accepted point is evaluated once" begin
        # x + αd can fall a rounding width inside a bound, so evaluating it in the line
        # search and then its projection in the main loop cost a second call at what is
        # the same point to within a few ulps.
        c = [2.6644647576513965, 2.4315480519908306, -0.8310801300270224, 2.191680369462489, -5.973033768404758]
        w = [27.99821012602616, 4.509132243751541, 1.415297678256256, 2.1084253095721075, 2.4406944118321636]
        l = [-0.9396236707040456, -0.17291887058593405, -0.935081612015503, -1.082608005797607, -0.898426798441686]
        u = [0.4596704334760291, 0.5820859319453694, 0.39933808684831007, 0.41393038856074815, 0.10136811645040592]
        x0 = [0.10904983262356838, 0.08683263763958968, 0.2219384726137681, -0.7138648591312208, -0.47288496423442106]
        xs = Vector{Vector{Float64}}()
        f = x -> (push!(xs, copy(x)); sum(w .* (x .- c) .^ 2) + 0.1 * sum(x)^2)
        g! = (G, x) -> (G .= 2 .* w .* (x .- c) .+ 0.2 * sum(x); G)
        for ls in (Optim.HZAW(), Optim.MTLS())
            empty!(xs)
            res = optimize(f, g!, l, u, x0, LBFGSB(linesearch = ls))
            @test Optim.converged(res)
            @test feasible(Optim.minimizer(res), l, u)
            @test !any(i -> isapprox(xs[i], xs[i+1]; rtol = 1e-14), 1:length(xs)-1)
        end
    end

    @testset "a line search that does not move x is a failure" begin
        # A step that leaves x unchanged used to end the run on the zero change in x as
        # converged. It is a failed line search: with a memory the iteration is redone
        # from B = I, and with none the run stops unconverged.
        struct ZeroStep <: Optim.LineSearcher end
        Optim.find_steplength(::ZeroStep, φ, φ0, dφ0, c; αmax = Inf) = (zero(c), φ0, false)
        rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        res = optimize(rosen, [-2.0, -2.0], [2.0, 2.0], [-1.2, 1.0], LBFGSB(linesearch = ZeroStep()))
        @test !Optim.converged(res)
        @test res.termination_code == Optim.TerminationCode.FailedLinesearch
        @test Optim.minimizer(res) == [-1.2, 1.0]
    end

    @testset "More-Thuente shrinks a step that gives a non-finite value" begin
        # ϕ(α) = (α - 1)², not finite beyond α = 2; the first trial is α = 4
        ϕdϕ(α) = α <= 2 ? ((α - 1)^2, 2 * (α - 1)) : (Inf, NaN)
        α, fα, gα = Optim.more_thuente(ϕdϕ, 4.0, 1.0, -2.0)
        @test isfinite(α) && isfinite(fα)
        @test 0 < α <= 2
        @test fα <= 1.0 + 1e-4 * α * -2.0
    end

    @testset "a NaN slope restarts the memory" begin
        # Brown almost-linear from 10 x0: the compact representation overflows on the
        # third iteration and every component of the direction is NaN. The slope used
        # to pass the descent test, so MTLS threw its assertion and HZAW stepped to NaN.
        n = 100
        function brownal(x)
            s = sum(x)
            sum(i -> (x[i] + s - (n + 1))^2, 1:n-1) + (prod(x) - 1)^2
        end
        function brownal_g!(G, x)
            s = sum(x)
            r = [x[i] + s - (n + 1) for i in 1:n-1]
            p = prod(x)
            for k in 1:n
                G[k] = 2 * (k < n ? r[k] : 0.0) + 2 * sum(r) + 2 * (p - 1) * prod(x[j] for j in 1:n if j != k)
            end
            G
        end
        x0 = fill(5.0, n)
        for ls in (Optim.HZAW(), Optim.MTLS())
            res = optimize(brownal, brownal_g!, fill(-Inf, n), fill(Inf, n), x0, LBFGSB(linesearch = ls))
            @test isfinite(Optim.minimum(res))
            @test all(isfinite, Optim.minimizer(res))
            @test Optim.minimum(res) < 1e-10
        end
    end

    @testset "the projected gradient norm of a NaN gradient is NaN" begin
        @test isnan(Optim.pg_norm([0.0, 0.0], [NaN, 1.0], fill(-Inf, 2), fill(Inf, 2)))
    end
end
