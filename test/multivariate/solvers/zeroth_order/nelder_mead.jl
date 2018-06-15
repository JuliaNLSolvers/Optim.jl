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
