@testset "L-BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",)

    iteration_exceptions = (("Extended Powell", 2000),)

    run_optim_tests(
        LBFGS(),
        f_increase_exceptions = ("Extended Rosenbrock",),
        skip = skip,
        iteration_exceptions = iteration_exceptions,
        show_name = debug_printing,
    )
end
