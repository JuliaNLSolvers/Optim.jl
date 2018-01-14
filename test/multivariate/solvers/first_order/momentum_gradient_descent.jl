@testset "Momentum Gradient Descent" begin
    # TODO: check the skips and exceptions, maybe it's enough to increase number of iterations?
    skip = ("Rosenbrock", "Extended Powell", "Extended Rosenbrock",
            "Trigonometric", "Penalty Function I")
    run_optim_tests(MomentumGradientDescent(),
                    skip = skip,
                    convergence_exceptions = (("Large Polynomial",1),  ("Himmelblau",1),
                                              ("Powell", 1), ("Powell", 2),
                                              ("Fletcher-Powell", 1),("Fletcher-Powell", 2)),
                    minimizer_exceptions = (("Powell", 2),),
                    minimum_exceptions = (("Large Polynomial", 1), ("Large Polynomial", 2)),
                    f_increase_exceptions = ("Exponential", "Polynomial"),
                    show_name = debug_printing)
end
