@testset "Conjugate Gradient" begin
    # TODO: Investigate the exceptions (could be they just need more iterations?)
    # Test Optim.cg for all differentiable functions in OptimTestProblems.UnconstrainedProblems.examples
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",)
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
        objective(X, B) = sum((X.-B).^2)/2

        function objective_gradient!(G, X, B)
            for i = 1:length(G)
                G[i] = X[i]-B[i]
            end
        end

        srand(1)
        B = rand(2,2)
        results = Optim.optimize(X -> objective(X, B), (G, X) -> objective_gradient!(G, X, B), rand(2,2), ConjugateGradient())
        @test Optim.converged(results)
        @test Optim.minimum(results) < 1e-8
    end
end
