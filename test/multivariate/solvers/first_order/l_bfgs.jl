@testset "L-BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric", "Polynomial")

    iteration_exceptions = (("Extended Powell", 2000),)

    run_optim_tests(
        LBFGS(),
        skip = skip,
        iteration_exceptions = iteration_exceptions,
        show_name = debug_printing,
    )
end
