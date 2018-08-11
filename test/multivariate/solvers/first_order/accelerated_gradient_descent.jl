@testset "Accelerated Gradient Descent" begin
    f(x) = x[1]^4
    function g!(storage, x)
        storage[1] = 4 * x[1]^3
        return
    end

    initial_x = [1.0]
    options = Optim.Options(show_trace = debug_printing, allow_f_increases=true)
    results = Optim.optimize(f, g!, initial_x, AcceleratedGradientDescent(), options)
    @test norm(Optim.minimum(results)) < 1e-6
    @test summary(results) == "Accelerated Gradient Descent"

    # TODO: Check why skip problems fail
    skip = ("Trigonometric", "Large Polynomial", "Parabola", "Paraboloid Random Matrix",
            "Paraboloid Diagonal", "Extended Rosenbrock", "Penalty Function I", "Beale",
            "Extended Powell",
             )
    run_optim_tests(AcceleratedGradientDescent();
                    skip = skip,
                    convergence_exceptions = (("Rosenbrock", 1),("Rosenbrock", 2)),
                    minimum_exceptions = (("Rosenbrock", 2),),
                    minimizer_exceptions = (("Rosenbrock", 2),),
                    iteration_exceptions = (("Powell", 1100),
                                            ("Rosenbrock", 10000),
                                            ("Polynomial", 1500),
                                            ("Fletcher-Powell", 10000),
                                            ("Extended Powell", 8000)),
                    f_increase_exceptions = ("Hosaki", "Polynomial", "Powell", "Himmelblau",
                                             "Extended Powell", "Fletcher-Powell",
                                             "Quadratic Diagonal", "Rosenbrock"),
                    show_name=debug_printing)#,
                    #show_res = debug_printing)
end
