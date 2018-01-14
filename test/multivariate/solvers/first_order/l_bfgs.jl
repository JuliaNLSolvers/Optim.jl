@testset "L-BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",)

    run_optim_tests(LBFGS(),
                    skip=skip,
                    show_name = debug_printing)
end
