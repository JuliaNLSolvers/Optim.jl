@testset "Callbacks" begin
    problem = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]

    f = MVP.objective(problem)
    g! = MVP.gradient(problem)
    h! = MVP.hessian(problem)
    initial_x = problem.initial_x
    d2 = OnceDifferentiable(f, g!, initial_x)
    d3 = TwiceDifferentiable(f, g!, h!, initial_x)

    for method in (NelderMead(), SimulatedAnnealing())
        a = 0
        cb = _ -> begin
            a += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3, store_trace = true)
        res1 = optimize(f, initial_x, method, options)
        @test a == 1+res1.iterations

        b = 0
        cb = os -> begin
            b += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3)
        res2 = optimize(f, initial_x, method, options)
        @test b == 1+res2.iterations

        # Test early stopping by callbacks
        iteration = -1
        cb = _ -> begin
            return (iteration += 1) == 5
        end
        options = Optim.Options(callback = cb)
        res3 = optimize(f, zeros(2), NelderMead(), options)
        @test res3.iterations == 5
    end

    for method in (BFGS(), ConjugateGradient(), GradientDescent(), MomentumGradientDescent())
        a = 0
        cb = _ -> begin
            a += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3, store_trace = true)
        res1 = optimize(d2, initial_x, method, options)
        @test a == 1+res1.iterations

        b = 0
        cb = _ -> begin
            b += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3)
        res2 = optimize(d2, initial_x, method, options)
        @test b == 1+res2.iterations

        c = 0
        cb = _ -> begin
            c += 1
            false
        end
        options = Optim.Options(callback = cb)
        res2 = optimize(d2, initial_x, method, options)
        @test c == 1+res2.iterations
    end

    for method in (Newton(),)
        a = 0
        cb = _ -> begin
            a += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3, store_trace = true)
        res1 = optimize(d3, initial_x, method, options)
        @test a == 1+res1.iterations

        b = 0
        cb = _ -> begin
            b += 1
            false
        end
        options = Optim.Options(callback = cb, show_every = 3)
        res2 = optimize(d3, initial_x, method, options)
        @test b == 1+res2.iterations

        c = 0
        cb = _ -> begin
            c += 1
            false
        end
        options = Optim.Options(callback = cb)
        res2 = optimize(d3, initial_x, method, options)
        @test c == 1+res2.iterations
    end

    res = optimize(x -> x^2, -5, 5, callback = _ -> true)
    @test res.iterations == 0
end
