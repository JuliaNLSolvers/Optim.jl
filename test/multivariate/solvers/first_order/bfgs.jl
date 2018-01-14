@testset "BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",)
    run_optim_tests(BFGS(); convergence_exceptions = (("Polynomial",1),),
                    skip=skip,
                    show_name = debug_printing)
end
