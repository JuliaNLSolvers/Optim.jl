@testset "Nelder Mead" begin
    # Test Optim.nelder_mead for all functions except Large Polynomials in MultivariateProblems.UnconstrainedProblems.examples
    skip = ("Large Polynomial", "Extended Powell", "Quadratic Diagonal",
            "Extended Rosenbrock", "Paraboloid Diagonal", "Paraboloid Random Matrix",
            "Trigonometric", "Penalty Function I",)
    run_optim_tests(NelderMead(),
                    convergence_exceptions = (("Powell", 1)),
                    minimum_exceptions = (("Exponential", 1), ("Exponential", 2)),
                    skip = skip, show_name = debug_printing)

    # Test if the trace is correctly stored.
    prob = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
    res = Optim.optimize(MVP.objective(prob), prob.initial_x, method = NelderMead(), store_trace = true, extended_trace=true)
    @test ( length(unique(Optim.g_norm_trace(res))) != 1 || length(unique(Optim.f_trace(res))) != 1 ) && issorted(Optim.f_trace(res)[end:1])
    @test !(res.trace[1].metadata["centroid"] === res.trace[end].metadata["centroid"])
end

@testset "Nelder Mead specific traces" begin
  f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
  x0 = [0.0,0.0]
  res = optimize(f, x0, NelderMead(),Optim.Options(store_trace=true,trace_simplex=true,extended_trace=true,iterations=2,show_trace=true))

  @test Optim.centroid_trace(res)[1] == [0.0125, 0.0]
  @test Optim.centroid_trace(res)[end] == [0.021875000000000002, -0.00625]

  @test_broken Optim.simplex_trace(res)[1] == [[0.0, 0.0], [0.025, 0.0], [0.0, 0.025]]
  @test Optim.simplex_trace(res)[end] ==[[0.065625, -0.018750000000000003], [0.025, 0.0], [0.018750000000000003, -0.0125]]

  @test_broken Optim.simplex_value_trace(res)[1] == [1.0, 0.9506640624999999, 1.0625] 
  @test Optim.simplex_value_trace(res)[end] == [0.9262175083160399, 0.9506640624999999, 0.9793678283691405]
end
