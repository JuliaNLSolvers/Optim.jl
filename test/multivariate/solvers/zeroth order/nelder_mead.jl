@testset "Nelder Mead" begin
	# Test Optim.nelder_mead for all functions except Large Polynomials in Optim.UnconstrainedProblems.examples

	run_optim_tests(NelderMead(),
					convergence_exceptions = (("Powell", 1),),
					skip =  (("Large Polynomial"),))

    # Test if the trace is correctly stored.
    prob = Optim.UnconstrainedProblems.examples["Rosenbrock"]
    res = Optim.optimize(prob.f, prob.initial_x, method = NelderMead(), store_trace = true)
    @test ( length(unique(Optim.g_norm_trace(res))) != 1 || length(unique(Optim.f_trace(res))) != 1 ) && issorted(Optim.f_trace(res)[end:1])
end
