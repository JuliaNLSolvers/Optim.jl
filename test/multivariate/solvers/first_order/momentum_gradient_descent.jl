@testset "Momentum Gradient Descent" begin
    # TODO: check the skips and exceptions, maybe it's enough to increase number of iterations?
    skip = ("Rosenbrock", "Extended Powell", "Extended Rosenbrock",
            "Trigonometric", "Penalty Function I", "Beale")
    run_optim_tests(MomentumGradientDescent(),
                    skip = skip,
                    convergence_exceptions = (("Large Polynomial",1),  ("Himmelblau",1),
                                              ("Fletcher-Powell", 1),("Fletcher-Powell", 2),
                                              ("Powell", 1)),
                    minimum_exceptions = (("Large Polynomial", 1), ("Large Polynomial", 2)),
                    iteration_exceptions = (("Paraboloid Random Matrix", 10000),
                                            ("Paraboloid Diagonal", 10000),
                                            ("Powell", 10000)),
                    f_increase_exceptions = ("Exponential", "Polynomial",
                                             "Paraboloid Random Matrix", "Hosaki"),
                    show_name = debug_printing)
end
