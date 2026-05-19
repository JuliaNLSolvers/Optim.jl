@testset "BFGS" begin
    run_optim_tests(
        BFGS();
        show_name = debug_printing,
    )
end
