using Optim, Test, Distributions, Random, LinearAlgebra
Random.seed!(3288)
@testset "Newton Trust Region" begin
    @testset "Subproblems I" begin
        # verify that solve_tr_subproblem! finds the minimum
        n = 2
        gr = [-0.74637, 0.52388]
        H = [0.945787 -3.07884; -3.07884 -1.27762]

        s = zeros(n)
        m, interior = Optim.solve_tr_subproblem!(gr, H, 1.0, s, max_iters = 100)

        for j = 1:10
            bad_s = rand(n)
            bad_s ./= norm(bad_s)  # boundary
            model(s2) = (gr'*s2)[] + 0.5 * (s2'*H*s2)[]
            @test model(s) <= model(bad_s) + 1e-8
        end
    end

    @testset "Subproblems II" begin
        # random Hessians--verify that solve_tr_subproblem! finds the minimum
        for i = 1:10000
            n = rand(1:10)
            gr = randn(n)
            H = randn(n, n)
            H += H'

            s = zeros(n)
            m, interior = Optim.solve_tr_subproblem!(gr, H, 1.0, s, max_iters = 100)

            model(s2) = (gr' * s2) + 0.5 * (s2' * H * s2)
            @test model(s) <= model(zeros(n)) + 1e-8  # origin

            for j = 1:10
                bad_s = rand(n)
                bad_s ./= norm(bad_s)  # boundary
                @test model(s) <= model(bad_s) + 1e-8
                bad_s .*= rand()  # interior
                @test model(s) <= model(bad_s) + 1e-8
            end
        end
    end

    @testset "Test problems" begin
        #######################################
        # First test the subproblem.
        Random.seed!(42)
        n = 5
        H = rand(n, n)
        H = H' * H + 4 * I
        H_eig = eigen(H)
        U = H_eig.vectors

        gr = zeros(n)
        gr[1] = 1.0
        s = zeros(Float64, n)

        true_s = -H \ gr
        s_norm2 = dot(true_s, true_s)
        true_m = dot(true_s, gr) + 0.5 * dot(true_s, H * true_s)

        # An interior solution
        delta = sqrt(s_norm2) + 1.0
        m, interior, lambda, hard_case, reached_solution =
            Optim.solve_tr_subproblem!(gr, H, delta, s)
        @test interior
        @test !hard_case
        @test reached_solution
        @test abs(m - true_m) < 1e-12
        @test norm(s - true_s) < 1e-12
        @test abs(lambda) < 1e-12

        # A boundary solution
        delta = 0.5 * sqrt(s_norm2)
        m, interior, lambda, hard_case, reached_solution =
            Optim.solve_tr_subproblem!(gr, H, delta, s)
        @test !interior
        @test !hard_case
        @test reached_solution
        @test m > true_m
        @test abs(norm(s) - delta) < 1e-12
        @test lambda > 0

        # A "hard case" where the gradient is orthogonal to the lowest eigenvector

        # Test the checking
        hard_case, lambda_index =
            Optim.check_hard_case_candidate([-1.0, 2.0, 3.0], [0.0, 1.0, 1.0], sqrt(eps()), sqrt(eps()))
        @test hard_case
        @test lambda_index == 2

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([-1.0, -1.0, 3.0], [0.0, 0.0, 1.0], sqrt(eps()), sqrt(eps()))
        @test hard_case
        @test lambda_index == 3

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], sqrt(eps()), sqrt(eps()))
        @test hard_case
        @test lambda_index == 4

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], sqrt(eps()), sqrt(eps()))
        @test !hard_case

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([-1.0, -1.0, -1.0], [0.0, 0.0, 1.0], sqrt(eps()), sqrt(eps()))
        @test !hard_case

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([-1.0, 2.0, 3.0], [1.0, 1.0, 1.0], sqrt(eps()), sqrt(eps()))
        @test !hard_case

        # Now check an actual hard case problem
        L = fill(0.1, n)
        L[1] = -1.0
        H = U * Matrix(Diagonal(L)) * U'
        H = 0.5 * (H' + H)
        @test issymmetric(H)
        gr = U[:, 2][:]
        @test abs(dot(gr, U[:, 1][:])) < 1e-12
        true_s = -H \ gr
        s_norm2 = dot(true_s, true_s)
        true_m = dot(true_s, gr) + 0.5 * dot(true_s, H * true_s)

        delta = 0.5 * sqrt(s_norm2)
        m, interior, lambda, hard_case, reached_solution =
            Optim.solve_tr_subproblem!(gr, H, delta, s)
        @test !interior
        @test hard_case
        @test reached_solution
        @test abs(lambda + L[1]) < 1e-4
        @test abs(norm(s) - delta) < 1e-12
        # The hard-case step must satisfy the boundary KKT system
        # (H + lambda*I)s = -gr; the reversed sign +gr also has norm delta
        # but is not the subproblem minimizer.
        @test norm((H + lambda * I) * s + gr) < 1e-10

        # An analytically solvable hard case: lambda = 2, p = [0, -1/3],
        # tau = sqrt(1 - 1/9), s = [±tau, -1/3], m = -7/6.
        H2 = Matrix(Diagonal([-2.0, 1.0]))
        gr2 = [0.0, 1.0]
        s2 = zeros(2)
        m2, interior2, lambda2, hard_case2, reached_solution2 =
            Optim.solve_tr_subproblem!(gr2, H2, 1.0, s2)
        @test hard_case2
        @test !interior2
        @test reached_solution2
        @test abs(lambda2 - 2.0) < 1e-12
        @test abs(norm(s2) - 1.0) < 1e-12
        @test abs(s2[2] + 1 / 3) < 1e-12
        @test abs(abs(s2[1]) - sqrt(8.0) / 3) < 1e-12
        @test abs(m2 - (-7 / 6)) < 1e-12


        #######################################
        # Next, test on actual optimization problems.

        function f(x::Vector)
            (x[1] - 5.0)^4
        end

        function g!(storage::Vector, x::Vector)
            storage[1] = 4.0 * (x[1] - 5.0)^3
        end

        function h!(storage::Matrix, x::Vector)
            storage[1, 1] = 12.0 * (x[1] - 5.0)^2
        end

        d = TwiceDifferentiable(f, g!, h!, [0.0])

        options =
            Optim.Options(store_trace = false, show_trace = false, extended_trace = true)
        results = Optim.optimize(d, [0.0], NewtonTrustRegion(), options)
        @test_throws ErrorException Optim.x_trace(results)
        @test length(results.trace) == 0
        @test Optim.g_converged(results)
        @test norm(Optim.minimizer(results) - [5.0]) < 0.01
        test_summary(results, "Newton's Method (Trust Region)")

        eta = 0.9

        function f_2(x::Vector)
            0.5 * (x[1]^2 + eta * x[2]^2)
        end

        function g!_2(storage::Vector, x::Vector)
            storage[1] = x[1]
            storage[2] = eta * x[2]
        end

        function h!_2(storage::Matrix, x::Vector)
            storage[1, 1] = 1.0
            storage[1, 2] = 0.0
            storage[2, 1] = 0.0
            storage[2, 2] = eta
        end

        d = TwiceDifferentiable(f_2, g!_2, h!_2, Float64[127, 921])

        results = Optim.optimize(d, Float64[127, 921], NewtonTrustRegion())
        @test Optim.g_converged(results)
        @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

        # Test Optim.newton for all twice differentiable functions in
        # MultivariateProblems.UnconstrainedProblems.examples
        @testset "Optim problems" begin
            run_optim_tests(
                NewtonTrustRegion();
                skip = ("Trigonometric",),
                show_name = debug_printing,
            )
        end
    end


    @testset "PR #341" begin
        # verify that no PosDef exception is thrown
        Optim.solve_tr_subproblem!([0, 1.0], [-1000 0; 0.0 -999], 1e-2, ones(2))
    end

    @testset "Handle Inf without erroring" begin
        o = optimize(
            TwiceDifferentiable(
                t -> rand(),
                (g, t) -> (g .= t .+ 10),
                (h, t) -> NaN * t * t',
                ones(10),
            ),
            ones(10),
            NewtonTrustRegion(),
        )
        @test !(Optim.f_converged(o) || Optim.g_converged(o) || Optim.x_converged(o))
    end

    @testset "delta_min" begin
        c =
            (t, Δ, D, ke) ->
                t < Δ ? -(exp(-ke * t) - 1) * D / (ke * Δ) :
                -(exp(-ke * Δ) - 1) * D / (ke * Δ) * exp(-ke * (t - Δ))

        ke₀ = 0.5
        D₀ = 100.0
        t₁ = 2.0
        ll =
            Δ -> begin
                sum(
                    map(
                        zip(
                            [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0],
                            [
                                19.90278833504542,
                                29.50697731718643,
                                42.106713695572836,
                                60.402701110755814,
                                72.78413106065605,
                                48.58414814304506,
                                36.134598474160484,
                                24.137636435583193,
                                3.2819695104173814,
                            ],
                        ),
                    ) do (t, y)
                        ct = c(t, Δ, D₀, ke₀)
                        return logpdf(Normal(ct, ct * 0.1), y)
                    end,
                )
            end

        @test_throws DomainError NewtonTrustRegion(delta_min = -1.0)
        @test iszero(NewtonTrustRegion().delta_min)

        res = Optim.optimize(
            t -> -ll(t[1]),
            [2.1],
            NewtonTrustRegion(),
            Optim.Options(show_trace = false, allow_f_increases = false, g_tol = 1e-5),
        )
        @test Optim.termination_code(res) == Optim.TerminationCode.NoXChange

        res = Optim.optimize(
            t -> -ll(t[1]),
            [2.1],
            NewtonTrustRegion(; delta_min = 1e-8),
            Optim.Options(show_trace = false, allow_f_increases = false, g_tol = 1e-5),
        )
        @test Optim.termination_code(res) == Optim.TerminationCode.SmallTrustRegionRadius
    end

    @testset "Singular Hessian in TR subproblem solve" begin
        H = [1.0 1.0; 1.0 1.0]   # positive-semidefinite, singular: eigenvalues 0 and 2
        g = [1.0, 1.0]           # gradient in image space of H
        s = fill(NaN, 2)
        m, interior, λ, hard_case, reached = Optim.solve_tr_subproblem!(g, H, 1.0, s)

        @test !hard_case
        @test all(isfinite, s)            # correctly update `s` to a finite value
        @test (H + λ*I)*s ≈ -g atol=1e-6  # solves trust region problem
        # Positive-semidefinite up to eigensolver noise: λ sits one ulp above
        # -min_H_ev, so the smallest eigenvalue of H + λI is zero to rounding.
        @test all(≥(-sqrt(eps())), eigvals(H + λ*I))
        # The minimizer is the interior pseudo-inverse step s = -[0.5, 0.5]
        # with ‖s‖ = 0.707 < delta = 1 and model value -0.5.
        @test reached
        @test interior
        @test m ≈ -0.5 atol = 1e-8
        @test s ≈ [-0.5, -0.5] atol = 1e-6
    end
    @testset "non-finite Hessian leaves a well-defined zero step" begin
        H = [1.0 0.0; 0.0 NaN]
        g = [1.0, 1.0]
        s = fill(NaN, 2)
        m, interior, λ, hard_case, reached = Optim.solve_tr_subproblem!(g, H, 1.0, s)
        @test m == Inf
        @test !reached
        @test all(iszero, s)
    end
    @testset "zero Hessian returns the exact linear-model solution" begin
        H = zeros(2, 2)
        g = [3.0, 4.0]
        s = fill(NaN, 2)
        m, interior, λ, hard_case, reached = Optim.solve_tr_subproblem!(g, H, 2.0, s)
        @test reached
        @test !interior
        @test s ≈ [-1.2, -1.6]        # -delta * g / ‖g‖
        @test m ≈ -10.0               # -delta * ‖g‖
        @test λ ≈ 2.5                 # ‖g‖ / delta

        s = fill(NaN, 2)
        m, interior, λ, hard_case, reached =
            Optim.solve_tr_subproblem!(zeros(2), H, 2.0, s)
        @test reached
        @test interior
        @test all(iszero, s)
        @test iszero(m)
    end

    @testset "zero gradient with an indefinite Hessian is the hard case" begin
        H = Matrix(Diagonal([-2.0, 1.0]))
        g = zeros(2)
        s = fill(NaN, 2)
        m, interior, λ, hard_case, reached = Optim.solve_tr_subproblem!(g, H, 1.0, s)
        @test hard_case
        @test reached
        @test abs(norm(s) - 1.0) < 1e-12   # step to the boundary along v₁
        @test m ≈ -1.0                     # 0.5 * λ_min * delta²
    end

    @testset "scaled and Float32 subproblems" begin
        # The classification must be invariant under H -> c*H, g -> c*g.
        H0 = [2.0 0.0; 0.0 1e-4]
        g0 = [1.0, 1.0]
        for c in (1e-6, 1e6)
            s = fill(NaN, 2)
            m, interior, λ, hard_case, reached =
                Optim.solve_tr_subproblem!(c .* g0, c .* H0, 1.0, s)
            s0 = fill(NaN, 2)
            m0, interior0, _, _, reached0 =
                Optim.solve_tr_subproblem!(g0, H0, 1.0, s0)
            @test interior == interior0
            @test reached == reached0
            @test s ≈ s0 atol = 1e-8
            @test m ≈ c * m0 rtol = 1e-8
        end

        H32 = Float32[2.0 0.0; 0.0 3.0]
        g32 = Float32[1.0, 1.0]
        s32 = zeros(Float32, 2)
        m32, interior32, λ32, hard32, reached32 =
            Optim.solve_tr_subproblem!(g32, H32, 5.0f0, s32)
        @test m32 isa Float32
        @test λ32 isa Float32
        @test reached32
        @test interior32
        @test s32 ≈ Float32[-0.5, -1/3]

        # A Float32 boundary solve must be able to report convergence: the
        # historical absolute tolerance of 1e-10 sat below eps(Float32).
        s32 = zeros(Float32, 2)
        m32, interior32, λ32, hard32, reached32 =
            Optim.solve_tr_subproblem!(g32, H32, 0.1f0, s32, max_iters = 100)
        @test !interior32
        @test reached32
        @test abs(norm(s32) - 0.1f0) < 1e-5
    end
    @testset "f_abstol and x_reltol terminate the solver" begin
        # A shifted flat quartic: near the start the objective barely changes,
        # so a loose f_abstol should stop the run long before g_abstol does.
        f(x) = (x[1] - 5.0)^4 + 1.0
        g!(G, x) = (G[1] = 4.0 * (x[1] - 5.0)^3; G)
        h!(H, x) = (H[1, 1] = 12.0 * (x[1] - 5.0)^2; H)
        d2 = TwiceDifferentiable(f, g!, h!, [0.0])
        res = Optim.optimize(d2, [0.0], NewtonTrustRegion(),
            Optim.Options(f_abstol = 1e-3, g_abstol = 0.0, iterations = 10_000))
        @test Optim.f_converged(res)
        # With f_abstol honored this stops at 11 iterations; ignoring it, the
        # run only ends at 30 when the f change rounds to exactly zero.
        @test Optim.iterations(res) <= 15

        d3 = TwiceDifferentiable(f, g!, h!, [0.0])
        res = Optim.optimize(d3, [0.0], NewtonTrustRegion(),
            Optim.Options(x_reltol = 1e-3, g_abstol = 0.0, iterations = 10_000))
        @test Optim.x_converged(res)
        @test Optim.iterations(res) < 10_000
    end
end
