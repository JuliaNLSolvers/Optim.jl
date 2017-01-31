@testset "Nelder Mead" begin
	# Test Optim.nelder_mead for all functions except Large Polynomials in Optim.UnconstrainedProblems.examples
	for (name, prob) in Optim.UnconstrainedProblems.examples
		res = Optim.optimize(prob.f, prob.initial_x, NelderMead(), Optim.Options(iterations = 10000))
		if name == "Powell"
			res = Optim.optimize(prob.f, prob.initial_x, NelderMead(), Optim.Options(g_tol = 1e-12))
		elseif name == "Large Polynomial"
			# TODO do this only when a "run all" flag checked
			# res = Optim.optimize(f_prob, prob.initial_x, method=NelderMead(initial_simplex = Optim.AffineSimplexer(1.,1.)), iterations = 450_000)
		end
		!(name == "Large Polynomial") && @test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
	end

    # Test if the trace is correctly stored.
    prob = Optim.UnconstrainedProblems.examples["Rosenbrock"]
    res = Optim.optimize(prob.f, prob.initial_x, method = NelderMead(), store_trace = true)
    @test ( length(unique(Optim.g_norm_trace(res))) != 1 || length(unique(Optim.f_trace(res))) != 1 ) && issorted(Optim.f_trace(res)[end:1])
end
