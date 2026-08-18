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
            Optim.check_hard_case_candidate([-1.0, 2.0, 3.0], [0.0, 1.0, 1.0])
        @test hard_case
        @test lambda_index == 2

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([-1.0, -1.0, 3.0], [0.0, 0.0, 1.0])
        @test hard_case
        @test lambda_index == 3

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([-1.0, -1.0, -1.0], [0.0, 0.0, 0.0])
        @test hard_case
        @test lambda_index == 4

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([1.0, 2.0, 3.0], [0.0, 1.0, 1.0])
        @test !hard_case

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([-1.0, -1.0, -1.0], [0.0, 0.0, 1.0])
        @test !hard_case

        hard_case, lambda_index =
            Optim.check_hard_case_candidate([-1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
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
        using Optim
        H = [1.0 1.0; 1.0 1.0]   # positive-semidefinite, singular: eigenvalues 0 and 2
        g = [1.0, 1.0]           # gradient in image space of H
        s = fill(NaN, 2)
        _, _, λ, hard_case, _ = Optim.solve_tr_subproblem!(g, H, 1.0, s)

        @test !hard_case
        @test all(isfinite, s)            # correctly update `s` to a finite value
        @test (H + λ*I)*s ≈ -g atol=1e-6  # solves trust region problem
        @test all(≥(0), eigvals(H + λ*I)) # positive-definite
    end
    @testset "factorization retries do not consume root-finding iterations" begin
        # The first Cholesky at lambda ~ lambda_lb fails for this singular H,
        # and the retry used to burn one of the five root-finding iterations:
        # reached_solution was false at the default max_iters and true at 6,
        # with an identical step either way.
        H = [1.0 1.0; 1.0 1.0]
        g = [1.0, 0.0]
        s = zeros(2)
        m, interior, λ, hard_case, reached = Optim.solve_tr_subproblem!(g, H, 1.0, s)
        @test reached
        @test !interior
        @test abs(norm(s) - 1.0) < 1e-8
        @test m ≈ -0.799038 atol = 1e-5
    end
end
