@testset "Momentum Gradient Descent" begin
    # TODO: check the skips and exceptions, maybe it's enough to increase number of iterations?
    skip = ("Rosenbrock", "Extended Powell", "Extended Rosenbrock",
            "Trigonometric", "Penalty Function I", "Beale","Paraboloid Random Matrix")
    run_optim_tests(MomentumGradientDescent(),
                    skip = skip,
                    convergence_exceptions = (("Fletcher-Powell", 1), ("Fletcher-Powell", 2),),
                    iteration_exceptions = (("Paraboloid Diagonal", 10000),
                                            ("Powell", 10000)),
                    f_increase_exceptions = ("Exponential", "Polynomial",
                                             "Paraboloid Random Matrix", "Hosaki"),
                    show_name = debug_printing)
end
