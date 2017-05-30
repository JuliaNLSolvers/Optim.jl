 @testset "Conjugate Gradient" begin
	# Test Optim.cg for all differentiable functions in Optim.UnconstrainedProblems.examples
	run_optim_tests(ConjugateGradient(),
                    convergence_exceptions = (("Powell", 1), ("Powell", 2), ("Polynomial", 1)),
                    f_increase_exceptions = (("Hosaki"),))

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
