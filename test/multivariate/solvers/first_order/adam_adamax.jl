@testset "$name" for (name, method, skip, alpha, alpha_scale) in (
    (
        "Adam",
        Adam,
        # TODO: Check why skip problems fail
        (
            "Large Polynomial",
            "Parabola",
            "Paraboloid Random Matrix",
            "Paraboloid Diagonal",
            "Penalty Function I",
            "Polynomial",
            "Powell",
            "Extended Powell",
            "Trigonometric",
            "Himmelblau",
            "Rosenbrock",
            "Extended Rosenbrock",
            "Quadratic Diagonal",
            "Beale",
            "Fletcher-Powell",
            "Exponential",
        ),
        1e-5,
        0.0001,
    ),
    (
        "AdaMax",
        AdaMax,
        # TODO: Check why skip problems fail
        (
            "Trigonometric",
            "Large Polynomial",
            "Parabola",
            "Paraboloid Random Matrix",
            "Paraboloid Diagonal",
            "Extended Rosenbrock",
            "Penalty Function I",
            "Beale",
            "Extended Powell",
            "Himmelblau",
            "Polynomial",
            "Powell",
            "Exponential",
        ),
        1e-4,
        0.002,
    ),
)
    f(x) = x[1]^4
    function g!(storage, x)
        storage[1] = 4 * x[1]^3
        return
    end

    initial_x = [1.0]
    options = Optim.Options(
        show_trace = debug_printing,
        allow_f_increases = true,
        iterations = 100_000,
    )
    results = Optim.optimize(f, g!, initial_x, method(), options)
    @test norm(Optim.minimum(results)) < 1e-6
    test_summary(results, name)

    run_optim_tests(method(); skip = skip, show_name = debug_printing)

    alpha_scheduler(iter) = alpha_scale * (1 + 0.99^iter)
    results = Optim.optimize(f, g!, initial_x, method(alpha = alpha_scheduler), options)
    @test norm(Optim.minimum(results)) < 1e-6
    test_summary(results, name)

    # verifying the alpha values over iterations and also testing extended_trace
    # this way we test both alpha scheduler and the working of
    # extended_trace=true option
    options = Optim.Options(
        show_trace = debug_printing,
        allow_f_increases = true,
        iterations = 1000,
        extended_trace = true,
        store_trace = true,
    )
    results = Optim.optimize(f, g!, initial_x, method(alpha = alpha), options)

    @test prod(
        map(
            iter -> results.trace[iter].metadata["Current step size"],
            2:results.iterations+1,
        ) .== alpha,
    )

    results = Optim.optimize(f, g!, initial_x, method(alpha = alpha_scheduler), options)

    @test map(
        iter -> results.trace[iter].metadata["Current step size"],
        2:results.iterations+1,
    ) == alpha_scheduler.(1:results.iterations)
end
