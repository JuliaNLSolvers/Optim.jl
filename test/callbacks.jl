let
    problem = Optim.UnconstrainedProblems.examples["Rosenbrock"]

    f = problem.f
    g! = problem.g!
    h! = problem.h!
    initial_x = problem.initial_x
    d2 = DifferentiableFunction(f, g!)
    d3 = TwiceDifferentiableFunction(f, g!, h!)

    for method in (NelderMead(), SimulatedAnnealing())
        ot_run = false
        cb = tr -> begin
            @test tr[end].iteration % 3 == 0
            ot_run = true
            false
        end
        options = Optim.Options(callback = cb, show_every=3, store_trace=true)
        optimize(f, initial_x, method, options)
        @test ot_run == true

        os_run = false
        cb = os -> begin
            @test os.iteration % 3 == 0
            os_run = true
            false
        end
        options = Optim.Options(callback = cb, show_every=3)
        optimize(f, initial_x, method, options)
        @test os_run == true

        # Test early stopping by callbacks
        options = Optim.Options(callback = x -> x.iteration == 5 ? true : false)
        optimize(f, zeros(2), NelderMead(), options)
    end

    for method in (BFGS(),
                   ConjugateGradient(),
                   GradientDescent(),
                   MomentumGradientDescent())
        ot_run = false
        cb = tr -> begin
            @test tr[end].iteration % 3 == 0
            ot_run = true
            false
        end
        options = Optim.Options(callback = cb, show_every=3, store_trace=true)
        optimize(d2, initial_x, method, options)
        @test ot_run == true

        os_run = false
        cb = os -> begin
            @test os.iteration % 3 == 0
            os_run = true
            false
        end
        options = Optim.Options(callback = cb, show_every=3)
        optimize(d2, initial_x, method, options)
        @test os_run == true
    end

    for method in (Newton(),)
        ot_run = false
        cb = tr -> begin
            @test tr[end].iteration % 3 == 0
            ot_run = true
            false
        end
        options = Optim.Options(callback = cb, show_every=3, store_trace=true)
        optimize(d3, initial_x, method, options)
        @test ot_run == true

        os_run = false
        cb = os -> begin
            @test os.iteration % 3 == 0
            os_run = true
            false
        end
        options = Optim.Options(callback = cb, show_every=3)
        optimize(d3, initial_x, method, options)
        @test os_run == true
    end
end
