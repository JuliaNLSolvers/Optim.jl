@testset "BFGS" begin
    # Trigonometric ends up in a local minimum
    skip = ("Trigonometric",)
    run_optim_tests(
        BFGS();
        skip,
        show_name = debug_printing,
    )
end
