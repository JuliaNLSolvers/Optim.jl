@testset "BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric", "Penalty Function I")

    iteration_exceptions = (("Extended Powell", 4000),)
    run_optim_tests(
        BFGS();
        convergence_exceptions = (("Polynomial", 1),),
        f_increase_exceptions = ("Extended Rosenbrock",),
        skip,
        iteration_exceptions,
        show_name = debug_printing,
    )
end
