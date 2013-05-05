using Optim

for (name, prob) in Optim.UnconstrainedProblems.examples
	if prob.isdifferentiable
		if name == "Himmelbrau"
			continue
		end
		df = DifferentiableFunction(prob.f, prob.g!)
		res = Optim.cg(df, prob.initial_x)
		if length(prob.solutions) == 1
			@assert norm(res.minimum - prob.solutions[1]) < 1e-2
		end
	end
end
