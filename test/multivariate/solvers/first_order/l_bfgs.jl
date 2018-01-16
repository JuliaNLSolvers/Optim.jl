@testset "L-BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",)

    run_optim_tests(LBFGS(),
                    f_increase_exceptions = ("Extended Rosenbrock",),
                    skip=skip,
                    show_name = debug_printing)
end
