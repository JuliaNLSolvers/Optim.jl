@testset "BFGS" begin
    run_optim_tests(BFGS(); convergence_exceptions = (("Polynomial",1),))
end
