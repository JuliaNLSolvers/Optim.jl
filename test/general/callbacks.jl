@testset "Callbacks" begin
    problem = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]

    f = MVP.objective(problem)
    g! = MVP.gradient(problem)
    h! = MVP.hessian(problem)
    initial_x = problem.initial_x
    d2 = OnceDifferentiable(f, g!, initial_x)
    d3 = TwiceDifferentiable(f, g!, h!, initial_x)

    # The callback is called for the initial state and after every iteration
    for (method, objective) in (
        (NelderMead(), f),
        (SimulatedAnnealing(), f),
        (BFGS(), d2),
        (ConjugateGradient(), d2),
        (GradientDescent(), d2),
        (MomentumGradientDescent(), d2),
        (Newton(), d3),
    )
        for options in ((; show_every = 3, store_trace = true), (; show_every = 3), (;))
            calls = 0
            cb = _ -> begin
                calls += 1
                false
            end
            res = optimize(objective, initial_x, method, Optim.Options(; callback = cb, options...))
            @test calls == 1 + res.iterations
        end
    end

    # Test early stopping by callbacks
    for method in (NelderMead(), SimulatedAnnealing())
        iteration = -1
        cb = _ -> begin
            return (iteration += 1) == 5
        end
        res = optimize(f, zeros(2), method, Optim.Options(callback = cb))
        @test res.iterations == 5
    end

    res = optimize(x -> x^2, -5, 5, callback = _ -> true)
    @test res.iterations == 0
end
