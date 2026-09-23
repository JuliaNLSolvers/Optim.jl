@testset "Feasibility (IPNewton drift)" begin
    # Regression scaffold (milestone M0) for the interior-point
    # feasibility-drift bug. On strongly-curved nonlinear inequality
    # constraints, IPNewton's backtracking line search can trade feasibility
    # of c(x) for a lower barrier objective: the slack variable stays
    # positive (so the barrier is satisfied) while c(x) drifts across the
    # constraint boundary, and the search then stalls at an infeasible,
    # non-stationary point. The proper fix is a filter line search plus a
    # feasibility-restoration phase (Wächter & Biegler 2006).
    #
    # `Optim.constraint_violation` (θ) and `Optim.barrier_objective` (ϕ) are
    # the two scalars such a filter operates on; this file tests those
    # primitives, guards the existing constrained problems against silently
    # becoming infeasible, and pins the known drift cases as `@test_broken`.
    # When the filter line search lands, the broken cases will report an
    # "Unexpected Pass" — that is the signal to promote them to `@test`.

    # Max violation of all constraints (box + nonlinear, ineq + eq) at x,
    # using the parsed sign convention σ(v - b) ≥ 0 for inequalities.
    function maxviol(cons, x)
        b = cons.bounds
        v = zero(eltype(x))
        for (i, σ, bb) in zip(b.ineqx, b.σx, b.bx)
            v = max(v, max(zero(v), σ * (bb - x[i])))
        end
        for (i, vv) in zip(b.eqx, b.valx)
            v = max(v, abs(x[i] - vv))
        end
        if b.nc > 0
            c = zeros(eltype(x), b.nc)
            cons.c!(c, x)
            for (i, σ, bb) in zip(b.ineqc, b.σc, b.bc)
                v = max(v, max(zero(v), σ * (bb - c[i])))
            end
            for (i, vv) in zip(b.eqc, b.valc)
                v = max(v, abs(c[i] - vv))
            end
        end
        v
    end

    @testset "θ/ϕ primitives" begin
        f = θ -> (θ[1] - 2)^2 + (4 * θ[2] + 3)^2
        c! = (c, θ) -> (c[1] = θ[2] + (θ[1] - 4)^2; c)
        x0 = [4.0, -0.5]   # interior: c = -0.5 < 0
        df = TwiceDifferentiable(f, x0)
        dfc = TwiceDifferentiableConstraints(c!, [-Inf, -Inf], [Inf, Inf], [-Inf], [0.0])
        st = Optim.initial_state(IPNewton(), Optim.Options(), df, dfc, x0)

        # θ is a true (nonnegative) norm and vanishes when the slack coupling
        # s = σ(c - b) is consistent, as it is at a freshly built state.
        θ = Optim.constraint_violation(dfc, st)
        @test θ ≥ 0
        @test θ ≤ 1e-8

        # ϕ = f(x) + barrier penalty (the part of the Lagrangian excluding the
        # equality term).
        ϕ = Optim.barrier_objective(dfc, st)
        @test isfinite(ϕ)
        @test ϕ == st.f_x + Optim.barrier_value(dfc.bounds, st)
    end

    @testset "fraction-to-the-boundary (M1)" begin
        # τⱼ = max(τ_min, 1 - μ), W&B eq. (8)
        @test Optim.fraction_to_boundary(0.0) == 1.0          # μ=0  → 1
        @test Optim.fraction_to_boundary(0.001) ≈ 0.999       # small μ → 1-μ
        @test Optim.fraction_to_boundary(0.5) == 0.99         # large μ → τ_min
        @test Optim.fraction_to_boundary(0.5; τ_min = 0.8) == 0.8
        for μ in (0.0, 1e-6, 0.01, 0.5, 5.0)
            τ = Optim.fraction_to_boundary(μ)
            @test 0 < τ ≤ 1                                   # always a valid fraction
        end
        @test Optim.fraction_to_boundary(0.1) isa Float64     # type stability

        # estimate_maxstep, W&B eq. (15): largest α with x + α s ≥ (1-τ) x.
        # τ = 1 (default) is the plain step-to-the-boundary rule.
        @test Optim.estimate_maxstep(Inf, [1.0], [-2.0]) == 0.5          # x/(-s) = 0.5
        @test Optim.estimate_maxstep(Inf, [1.0], [-2.0], 1.0) == 0.5     # explicit τ=1 matches
        @test Optim.estimate_maxstep(Inf, [1.0], [-2.0], 0.9) == 0.45    # τ-margin scales the bound
        @test Optim.estimate_maxstep(Inf, [1.0], [2.0]) == Inf           # s ≥ 0 imposes no bound
        @test Optim.estimate_maxstep(1.0, [1.0], [-2.0]) == 0.5          # honors incoming αmax cap
        @test Optim.estimate_maxstep(Inf, [1.0, 1.0], [-2.0, -10.0]) == 0.1  # min over components

        # the binding component lands exactly on the (1-τ) margin
        τ = 0.9
        x = [2.0, 5.0]
        s = [-4.0, 1.0]
        α = Optim.estimate_maxstep(Inf, x, s, τ)
        @test x[1] + α * s[1] ≈ (1 - τ) * x[1]
        @test x[2] + α * s[2] ≥ (1 - τ) * x[2]
    end

    @testset "barrier trial quantities (M2)" begin
        f = θ -> (θ[1] - 2)^2 + (4 * θ[2] + 3)^2
        c! = (c, θ) -> (c[1] = θ[2] + (θ[1] - 4)^2; c)
        x0 = [4.0, -0.5]
        df = TwiceDifferentiable(f, x0)
        dfc = TwiceDifferentiableConstraints(c!, [-Inf, -Inf], [Inf, Inf], [-Inf], [0.0])
        st = Optim.initial_state(IPNewton(), Optim.Options(), df, dfc, x0)
        # initial_state leaves state.s as NaN (it is filled by solve_step!
        # before the line search runs); set a finite primal direction so the
        # trial updates below are well defined.
        st.s .= [0.01, -0.02]
        st.bstep.slack_c .= [0.03]

        # At α = 0 the trial evaluator reproduces the current-point primitives.
        ϕ0, θ0 = Optim._barrier_linesearch(0.0, df, dfc, st)
        @test ϕ0 ≈ Optim.barrier_objective(dfc, st)
        @test θ0 ≈ Optim.constraint_violation(dfc, st)

        # ∇ϕᵀd matches a finite difference of ϕ along the chosen primal direction.
        slope = Optim.barrier_objective_slope(dfc, st)
        h = 1e-6
        ϕh, _ = Optim._barrier_linesearch(h, df, dfc, st)
        ϕ0b, _ = Optim._barrier_linesearch(0.0, df, dfc, st)
        @test (ϕh - ϕ0b) / h ≈ slope rtol = 1e-4

        # ϕ excludes the equality (λ) term: ϕ = L_x - ev at the current point.
        Optim.update_fgh!(df, dfc, st, IPNewton())
        @test Optim.barrier_objective(dfc, st) ≈ st.L_x - st.ev
    end

    @testset "filter predicates (M3)" begin
        p = Optim.FilterParams{Float64}()
        @test p.γ_θ == 1e-5
        @test p.γ_ϕ == 1e-5
        @test p.δ == 1.0
        @test p.s_θ == 1.1
        @test p.s_ϕ == 2.3
        @test p.η_ϕ == 1e-4
        @test p.γ_α == 0.05
        @test p.τ_min == 0.99

        # Filter membership (eqs. 21-22).
        F = Optim.Filter(10.0)                       # θ_max = 10, no corners yet
        @test Optim.filter_acceptable(F, 1.0, 5.0)   # empty filter, below θ_max
        @test !Optim.filter_acceptable(F, 10.0, -100.0)  # θ ≥ θ_max is blocked
        Optim.augment_filter!(F, 2.0, 3.0, p)        # corner ((1-γθ)·2, 3-γϕ·2)
        @test F.θ[end] ≈ (1 - p.γ_θ) * 2.0
        @test F.ϕ[end] ≈ 3.0 - p.γ_ϕ * 2.0
        @test !Optim.filter_acceptable(F, 2.0, 3.0)  # dominated by the corner
        @test Optim.filter_acceptable(F, 1.5, 3.0)   # better θ escapes the corner
        @test Optim.filter_acceptable(F, 2.0, 2.9)   # better ϕ escapes the corner

        # Sufficient decrease (eq. 18): θ-progress or ϕ-progress vs (θ_k, ϕ_k).
        @test Optim.sufficient_decrease(0.9, 5.0, 1.0, 5.0, p)
        @test Optim.sufficient_decrease(1.0, 4.9, 1.0, 5.0, p)
        @test !Optim.sufficient_decrease(1.0, 5.0, 1.0, 5.0, p)

        # Switching condition (eq. 19): needs descent and a dominant ϕ term.
        @test Optim.switching_condition(-1.0, 1.0, 0.0, p)
        @test !Optim.switching_condition(1.0, 1.0, 0.0, p)      # not a descent direction
        @test !Optim.switching_condition(-1.0, 1e-3, 10.0, p)   # θ term dominates

        # Armijo on ϕ (eq. 20).
        @test Optim.armijo_barrier(5.0 - 1e-3, 5.0, -2.0, 0.5, p)
        @test !Optim.armijo_barrier(5.0, 5.0, -2.0, 0.5, p)

        # Minimum step (eq. 23): three branches, all positive.
        θ_min = 1e-4
        @test Optim.filter_min_step(1e-5, -1.0, θ_min, p) > 0       # ∇ϕ<0, θ_k ≤ θ_min
        @test Optim.filter_min_step(1.0, -1.0, θ_min, p) > 0        # ∇ϕ<0, θ_k > θ_min
        @test Optim.filter_min_step(1.0, 1.0, θ_min, p) == p.γ_α * p.γ_θ  # ∇ϕ ≥ 0
    end

    @testset "existing constrained problems stay feasible" begin
        # None of the registered problems drift today (worst residual ≈ 1e-9);
        # this guard fails loudly if a future change reintroduces infeasibility.
        ftol = 1e-5
        method = IPNewton()
        for (name, prob) in MVP.ConstrainedProblems.examples
            df = TwiceDifferentiable(
                MVP.objective(prob),
                MVP.gradient(prob),
                MVP.objective_gradient(prob),
                MVP.hessian(prob),
                prob.initial_x,
            )
            cd = prob.constraintdata
            cons = TwiceDifferentiableConstraints(
                cd.c!,
                cd.jacobian!,
                cd.h!,
                cd.lx,
                cd.ux,
                cd.lc,
                cd.uc,
            )
            res = optimize(
                df,
                cons,
                prob.initial_x,
                method,
                Optim.Options(; Optim.default_options(method)...),
            )
            @test maxviol(cons, Optim.minimizer(res)) ≤ ftol
        end
    end

    @testset "drift cases (known broken until filter line search)" begin
        ftol = 1e-6
        # (name, f, c!, lc, uc, x0). Each is feasible at x0 with a strongly
        # curved active inequality; the unconstrained optimum lies outside the
        # feasible region so the constraint is active at the true optimum.
        drift_problems = [
            (
                "curved-k4",  # the originally reported case
                θ -> (θ[1] - 2)^2 + (4 * θ[2] + 3)^2,
                (c, θ) -> (c[1] = θ[2] + (θ[1] - 4)^2; c),
            ),
            (
                "curved-k6",  # steeper objective in y → stronger drift
                θ -> (θ[1] - 2)^2 + (6 * θ[2] + 3)^2,
                (c, θ) -> (c[1] = θ[2] + (θ[1] - 4)^2; c),
            ),
        ]
        x0 = [4.0, -0.5]
        for (name, f, c!) in drift_problems
            df = TwiceDifferentiable(f, x0)
            dfc = TwiceDifferentiableConstraints(c!, [-Inf, -Inf], [Inf, Inf], [-Inf], [0.0])
            res = optimize(df, dfc, x0, IPNewton())
            x = Optim.minimizer(res)
            # The returned point is currently infeasible and the solver stalls,
            # so neither of these holds yet. They flip once the filter line
            # search keeps the iterates feasible.
            @test_broken maxviol(dfc, x) ≤ ftol
            @test_broken Optim.converged(res)
        end
    end
end
