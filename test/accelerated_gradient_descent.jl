@testset "Accelerated Gradient Descent" begin
    f(x) = x[1]^4
    function g!(storage, x)
        storage[1] = 4 * x[1]^3
        return
    end

    initial_x = [1.0]
    options = Optim.Options(show_trace = true, allow_f_increases=true)
    results = Optim.optimize(f, g!, initial_x, AcceleratedGradientDescent(), options)
    @test norm(Optim.minimum(results)) < 1e-6
    @test summary(results) == "Accelerated Gradient Descent"

    run_optim_tests(AcceleratedGradientDescent(); skip = ("Large Polynomial","Parabola"),
                                                  convergence_exceptions = (("Rosenbrock", 1),("Rosenbrock", 2)),
                                                  iteration_exceptions = (("Powell", 1100),
                                                                          ("Polynomial", 1500)))
end
