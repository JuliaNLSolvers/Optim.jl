@testset "Momentum Gradient Descent" begin
    run_optim_tests(MomentumGradientDescent(),
                    skip = ("Powell", "Rosenbrock"),
                    convergence_exceptions = (("Large Polynomial",1),
                     ("Himmelblau",1), ("Powell", 1)),
                    minimizer_exceptions = (("Powell", 2),),
                    f_increase_exceptions = ("Exponential", "Polynomial"))
end
