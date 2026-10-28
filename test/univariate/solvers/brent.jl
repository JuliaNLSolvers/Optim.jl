@testset "Brent's Method" begin
    for (name, prob) in OptimTestProblems.UnivariateProblems.examples
        for T in (Float64, BigFloat)
            results = optimize(prob.f, convert(Array{T}, prob.bounds)..., method = Brent())

            @test Optim.converged(results)
            @test norm(Optim.minimizer(results) .- prob.minimizers) < 1e-7
        end
    end

    ## corner cases - empty and zero-width brackets
    result = optimize(sqrt, 4.0, 4.0, method = Brent())
    @test Optim.converged(result)
    @test Optim.minimizer(result) == 4.0
    @test Optim.minimum(result) == 2.0
    @test_throws ErrorException optimize(identity, 2.0, 1.0, Brent())
    test_summary(result, "Brent's Method")

    ## corner cases - largely flat functions
    result = optimize(x -> sign(x), -2, 2)
    @test Optim.converged(result)
    @test Optim.minimum(result) == -1.0
    result = optimize(x -> sign(x), -1, 2)
    @test Optim.converged(result)
    @test Optim.minimum(result) == -1.0
    result = optimize(x -> sign(x), -2, 1)
    @test Optim.converged(result)
    @test Optim.minimum(result) == -1.0
    ## the first interior point sits on a plateau above one of the bounds (#828)
    for (a, b) in ((-2.0, 2.0), (-1.0, 2.0), (-2.0, 1.0))
        result = optimize(x -> -sign(x), a, b, Brent())
        @test Optim.minimum(result) == -1.0
    end
    result = optimize(x -> x > 1 ? 2.0 : x^2, -1.0, 10.0, Brent())
    @test Optim.minimum(result) < 1e-10
    @test abs(Optim.minimizer(result)) < 1e-5

    ## the parabolic step is taken whenever it lands inside the bracket; with the
    ## bracket test the wrong way round this run took 24 evaluations, mostly golden
    ## section steps
    f_exp(x) = exp(0.2x) - 0.2 * exp(0.1) * x
    result = optimize(f_exp, 0.0, 1.0, Brent(evaluate_bounds = false))
    @test Optim.minimizer(result) ≈ 0.5 atol = 1e-7
    @test Optim.f_calls(result) <= 12
    ## evaluating the bounds costs two calls when they do not move the search
    @test Optim.f_calls(optimize(f_exp, 0.0, 1.0, Brent())) == Optim.f_calls(result) + 2

    ## time limit
    function slow_obj(x)
        sleep(0.05)
        return sin(x)
    end
    result = optimize(x -> slow_obj(x), 0, 2π, Brent(); time_limit = 0.01)
    @test result.f_calls == 3
    result = optimize(x -> slow_obj(x), 0, 2π, Brent(evaluate_bounds = false); time_limit = 0.01)
    @test result.f_calls == 1

    result = Optim.optimize(
        x -> sin(x),
        0,
        2π,
        Optim.Brent();
        abs_tol = 1e-4,
        store_trace = false,
        show_trace = true,
        iterations = 2,
    )
end
