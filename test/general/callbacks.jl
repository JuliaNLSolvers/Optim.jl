@testset "Callbacks" begin
    problem = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]

    f = MVP.objective(problem)
    g! = MVP.gradient(problem)
    h! = MVP.hessian(problem)
    initial_x = problem.initial_x
    d2 = OnceDifferentiable(f, g!, initial_x)
    d3 = TwiceDifferentiable(f, g!, h!, initial_x)

    for method in (NelderMead(), SimulatedAnnealing())
        ot_count = 0
        cb = tr -> begin
            ot_count += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3, store_trace = true)
        res1 = optimize(f, initial_x, method, options)
        @test ot_count == 1+res1.iterations

        os_count_2 = 0
        cb = os -> begin
            os_count_2 += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3)
        res2 = optimize(f, initial_x, method, options)
        @test os_count_2 == 1+res2.iterations

        # Test early stopping by callbacks
        options = Optim.Options(callback = x -> x.iteration == 5 ? true : false)
        res3 = optimize(f, zeros(2), NelderMead(), options)
        @test res3.iterations == 5
    end

    for method in (BFGS(), ConjugateGradient(), GradientDescent(), MomentumGradientDescent())
        ot_count = 0
        cb = tr -> begin
            ot_count += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3, store_trace = true)
        res1 = optimize(d2, initial_x, method, options)
        @test ot_count == 1+res1.iterations

        os_count = 0
        cb = os -> begin
            os_count += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3)
        res2 = optimize(d2, initial_x, method, options)
        @test os_count == 1+res2.iterations
    end

    for method in (Newton(),)
        ot_count = 0
        cb = tr -> begin
            ot_count += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3, store_trace = true)
        res1 = optimize(d3, initial_x, method, options)
        @test ot_count == 1+res1.iterations

        os_count = 0
        cb = os -> begin
            os_count += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3)
        res2 = optimize(d3, initial_x, method, options)
        @test os_count == 1+res2.iterations
    end

    res = optimize(x -> x^2, -5, 5, callback = _ -> true)
    @test res.iterations == 0
end
