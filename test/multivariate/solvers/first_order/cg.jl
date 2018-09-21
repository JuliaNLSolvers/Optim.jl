@testset "Conjugate Gradient" begin
    # TODO: Investigate the exceptions (could be they just need more iterations?)
    # Test Optim.cg for all differentiable functions in MultivariateProblems.UnconstrainedProblems.examples
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric", "Extended Powell")
    run_optim_tests(ConjugateGradient(),
                    skip=skip,
                    convergence_exceptions = (("Powell", 1), ("Powell", 2), ("Polynomial", 1),
                                              ("Extended Rosenbrock", 1),
                                              ("Extended Powell", 1),
                                              ("Extended Powell", 2)),
                    minimum_exceptions = (("Paraboloid Diagonal", 1)),
                    minimizer_exceptions = (("Paraboloid Diagonal", 1),
                                            ("Extended Powell", 1),
                                            ("Extended Powell", 2)),
                    f_increase_exceptions = (("Hosaki"),),
                    iteration_exceptions = (("Paraboloid Diagonal", 10000),),
                    show_name = debug_printing)

    @testset "matrix input" begin
        cg_objective(X, B) = sum(abs2, X .- B)/2

        function cg_objective_gradient!(G, X, B)
            for i = 1:length(G)
                G[i] = X[i]-B[i]
            end
        end

        Random.seed!(1)
        B = rand(2,2)
        results = Optim.optimize(X -> cg_objective(X, B), (G, X) -> cg_objective_gradient!(G, X, B), rand(2,2), ConjugateGradient())
        @test Optim.converged(results)
        @test Optim.minimum(results) < 1e-8
    end

    @testset "Undefined beta_k behaviour" begin
        # Ref #669
        # If y_k is zero, then betak is undefined.
        # Check that we don't produce NaNs

        f(x) = 100 - x[1] + exp(x[1] - 100)
        g!(grad, x) = grad[1] = -1 + exp(x[1] - 100)
        res = optimize(f, g!, [0.0], ConjugateGradient(alphaguess=LineSearches.InitialStatic(alpha=1.0), linesearch=LineSearches.BackTracking()))
        @test Optim.converged(res)
        @test Optim.minimum(res) ≈ 1.0
    end
end
