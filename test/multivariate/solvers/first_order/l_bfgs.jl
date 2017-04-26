@testset "L-BFGS" begin
    run_optim_tests(LBFGS(), convergence_exceptions = (("Rosenbrock", 1), ("Polynomial", 1)))
end
