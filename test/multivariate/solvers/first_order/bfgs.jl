@testset "BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    # Penalty Function I gives an increase in function value of a runner which is weird given the resets
    skip = ("Trigonometric", "Penalty Function I")

    iteration_exceptions = (("Extended Powell", 4000),)
    run_optim_tests(
        BFGS();
        skip,
        iteration_exceptions,
        show_name = debug_printing,
    )
end
