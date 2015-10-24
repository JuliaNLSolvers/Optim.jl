using Optim

# Test Optim.cg for all differentiable functions in Optim.UnconstrainedProblems.examples
for (name, prob) in Optim.UnconstrainedProblems.examples
	if prob.isdifferentiable
		df = DifferentiableFunction(prob.f, prob.g!)
		res = Optim.optimize(df, prob.initial_x, method=ConjugateGradient())
			@assert norm(res.minimum - prob.solutions) < 1e-2
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
df = Optim.DifferentiableFunction(X -> objective(X, B), (X, G) -> objective_gradient!(X, G, B))
results = Optim.optimize(df, rand(2,2), method=ConjugateGradient())
@assert Optim.converged(results)
@assert results.f_minimum < 1e-8
end
