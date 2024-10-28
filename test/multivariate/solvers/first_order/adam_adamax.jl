@testset "Adam" begin
    f(x) = x[1]^4
    function g!(storage, x)
        storage[1] = 4 * x[1]^3
        return
    end

    initial_x = [1.0]
    options = Optim.Options(show_trace = debug_printing, allow_f_increases=true, iterations=100_000)
    results = Optim.optimize(f, g!, initial_x, Adam(), options)
    @test norm(Optim.minimum(results)) < 1e-6
    @test summary(results) == "Adam"

    # TODO: Check why skip problems fail
    skip = ("Large Polynomial", "Parabola", "Paraboloid Random Matrix",
            "Paraboloid Diagonal", "Penalty Function I", "Polynomial", "Powell",
             "Extended Powell", "Trigonometric", "Himmelblau", "Rosenbrock", "Extended Rosenbrock",
             "Quadratic Diagonal", "Beale", "Fletcher-Powell", "Exponential",
             )
    run_optim_tests(Adam();
                    skip = skip,
                    show_name = debug_printing)
end
@testset "AdaMax" begin
    f(x) = x[1]^4
    function g!(storage, x)
        storage[1] = 4 * x[1]^3
        return
    end

    initial_x = [1.0]
    options = Optim.Options(show_trace = debug_printing, allow_f_increases=true, iterations=100_000)
    results = Optim.optimize(f, g!, initial_x, AdaMax(), options)
    @test norm(Optim.minimum(results)) < 1e-6
    @test summary(results) == "AdaMax"

    # TODO: Check why skip problems fail
    skip = ("Trigonometric", "Large Polynomial", "Parabola", "Paraboloid Random Matrix",
            "Paraboloid Diagonal", "Extended Rosenbrock", "Penalty Function I", "Beale",
            "Extended Powell", "Himmelblau", "Large Polynomial", "Polynomial", "Powell",
            "Exponential",
             )
    run_optim_tests(AdaMax();
                    skip = skip,
                    show_name=debug_printing,
                    iteration_exceptions = (("Trigonometric", 1_000_000,),))
end
