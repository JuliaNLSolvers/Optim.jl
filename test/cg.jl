@testset "Conjugate Gradient" begin
	# Test Optim.cg for all differentiable functions in Optim.UnconstrainedProblems.examples
	for (name, prob) in Optim.UnconstrainedProblems.examples
		if prob.isdifferentiable
			df = OnceDifferentiable(prob.f, prob.g!)
			res = Optim.optimize(df, prob.initial_x, ConjugateGradient())
				@test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
		end
	end

	let
	objective(X, B) = sum((X.-B).^2)/2

	function objective_gradient!(X, G, B)
	    for i = 1:length(G)
	        G[i] = X[i]-B[i]
	    end
	end

	srand(1)
	B = rand(2,2)
	df = Optim.OnceDifferentiable(X -> objective(X, B), (X, G) -> objective_gradient!(X, G, B))
	results = Optim.optimize(df, rand(2,2), ConjugateGradient())
	@test Optim.converged(results)
	@test Optim.minimum(results) < 1e-8
	end
end
