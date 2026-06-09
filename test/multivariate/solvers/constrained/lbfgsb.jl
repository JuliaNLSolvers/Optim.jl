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
            rf = optimize(rosen, l, u, x0, Fminbox(LBFGS()))
            @test Optim.minimizer(rb) ≈ [1.0, 1.0] atol = 1e-4
            @test Optim.minimizer(rb) ≈ Optim.minimizer(rf) atol = 1e-3
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
end
