@testset "BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",)

    iteration_exceptions = (("Extended Powell", 4000),)
    run_optim_tests(
        BFGS();
        skip,
        iteration_exceptions,
        show_name = debug_printing,
    )
end
