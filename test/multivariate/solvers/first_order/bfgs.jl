@testset "BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",
            "Penalty Function I ") # technically only fails on windows...
    run_optim_tests(
        BFGS();
        convergence_exceptions = (("Polynomial", 1),),
        f_increase_exceptions = ("Extended Rosenbrock",),
        skip = skip,
        show_name = debug_printing,
    )
end
