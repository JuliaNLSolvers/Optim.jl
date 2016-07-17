using Optim

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

function f_nm(x::Vector)
  (100.0 - x[1])^2 + x[2]^2
end

function rosenbrock_nm(x::Vector)
  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

initial_x = [0.0, 0.0]

results = Optim.optimize(f_nm, initial_x, method=NelderMead())

@assert Optim.g_converged(results)
@assert norm(Optim.minimizer(results) - [100.0, 0.0]) < 0.01
@test_throws ErrorException Optim.x_trace(results)

results = Optim.optimize(rosenbrock_nm, initial_x, method=NelderMead())

@assert Optim.g_converged(results)
@assert norm(Optim.minimizer(results) - [1.0, 1.0]) < 0.01
@test_throws ErrorException Optim.x_trace(results)
