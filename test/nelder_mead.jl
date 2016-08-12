let
	# Test Optim.nelder_mead for all functions except Large Polynomials in Optim.UnconstrainedProblems.examples
	for (name, prob) in Optim.UnconstrainedProblems.examples
		f_prob = prob.f
		res = Optim.optimize(f_prob, prob.initial_x, NelderMead(), OptimizationOptions(iterations = 10000))
		if name == "Powell"
			res = Optim.optimize(f_prob, prob.initial_x, method=NelderMead(), g_tol = 1e-12)
		elseif name == "Large Polynomial"
			res = Optim.optimize(f_prob, prob.initial_x, method=NelderMead(initial_simplex = Optim.AffineSimplexer(1.,1.)), iterations = 500_000)
		end
		@assert norm(res.minimum - prob.solutions) < 1e-2
	end
end


# Test that deprecated syntax runs
let
	dep_prob = Optim.UnconstrainedProblems.examples["Rosenbrock"]
	optimize(dep_prob.f, dep_prob.initial_x, NelderMead(a = 1.0))
	optimize(dep_prob.f, dep_prob.initial_x, NelderMead(g = 2.0))
	optimize(dep_prob.f, dep_prob.initial_x, NelderMead(b = 0.5))
	optimize(dep_prob.f, dep_prob.initial_x, NelderMead(initial_simplex = Optim.AffineSimplexer()))
	optimize(dep_prob.f, dep_prob.initial_x, NelderMead(parameters = Optim.AdaptiveParameters()))
end
