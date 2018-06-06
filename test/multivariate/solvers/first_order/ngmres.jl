## REMEMBER TO UPDATE TESTS FOR BOTH THE N-GMRES and the O-ACCEL TEST SETS

@testset "N-GMRES" begin
    method = NGMRES
    solver = method()

    skip = ("Trigonometric", )
    run_optim_tests(solver; skip = skip,
                    iteration_exceptions = (("Penalty Function I", 10000), ),
                    show_name = debug_printing)

    # Specialized tests
    prob = MVP.UnconstrainedProblems.examples["Rosenbrock"]
    df = OnceDifferentiable(MVP.objective(prob),
                            MVP.gradient(prob),
                            prob.initial_x)

    @test solver.nlpreconopts.iterations == 1
    @test solver.nlpreconopts.allow_f_increases == true

    defopts = Optim.default_options(solver)
    @test defopts == Dict(:allow_f_increases => true)

    state = Optim.initial_state(solver, Optim.Options(;defopts...), df,
                                prob.initial_x)
    @test state.x          === state.nlpreconstate.x
    @test state.x_previous === state.nlpreconstate.x_previous
    @test size(state.X)    == (length(state.x), solver.wmax)
    @test size(state.R)    == (length(state.x), solver.wmax)
    @test size(state.Q)    == (solver.wmax, solver.wmax)
    @test size(state.ξ)    == (solver.wmax,)
    @test state.curw       == 1
    @test size(state.A)    == (solver.wmax, solver.wmax)
    @test length(state.b)  == solver.wmax
    @test length(state.xA) == length(state.x)

    # Test that tracing doesn't throw errors
    res = optimize(df, prob.initial_x, solver,
                   Optim.Options(extended_trace=true, store_trace=true;
                                 defopts...))

    @test Optim.converged(res)
    # The bounds are due to different systems behaving differently
    # TODO: is it a bad idea to hardcode these?
    @test 64 < Optim.iterations(res) < 84
    @test 234 < Optim.f_calls(res) <  286
    @test 234 < Optim.g_calls(res) < 286
    @test Optim.minimum(res) < 1e-10

    @test_throws AssertionError method(manifold=Optim.Sphere(), nlprecon = GradientDescent())

    for nlprec in (LBFGS, BFGS)
        solver = method(nlprecon=nlprec())
        clear!(df)
        res = optimize(df, prob.initial_x, solver)

        if !Optim.converged(res)
            display(res)
        end
        @test Optim.converged(res)
        @test Optim.minimum(res) < 1e-10
    end

    # O-ACCEL handles the InitialConstantChange functionality in a special way,
    # so we should test that it works well.
    for nlprec in (GradientDescent(),
                   GradientDescent(alphaguess = LineSearches.InitialConstantChange()))
        solver = method(nlprecon = nlprec,
                        alphaguess = LineSearches.InitialConstantChange())
        clear!(df)

        res = optimize(df, prob.initial_x, solver)

        if !Optim.converged(res)
            display(res)
        end
        @test Optim.converged(res)
        @test Optim.minimum(res) < 1e-10
    end
end

@testset "O-ACCEL" begin
    method = OACCEL
    solver = method()
    skip = ("Trigonometric", )
    run_optim_tests(solver; skip = skip,
                    iteration_exceptions = (("Penalty Function I", 10000), ),
                    show_name = debug_printing)

    prob = MVP.UnconstrainedProblems.examples["Rosenbrock"]
    df = OnceDifferentiable(MVP.objective(prob),
                            MVP.gradient(prob),
                            prob.initial_x)
    @test solver.nlpreconopts.iterations == 1
    @test solver.nlpreconopts.allow_f_increases == true

    defopts = Optim.default_options(solver)
    @test defopts == Dict(:allow_f_increases => true)

    state = Optim.initial_state(solver, Optim.Options(;defopts...), df,
                                prob.initial_x)
    @test state.x          === state.nlpreconstate.x
    @test state.x_previous === state.nlpreconstate.x_previous
    @test size(state.X)    == (length(state.x), solver.wmax)
    @test size(state.R)    == (length(state.x), solver.wmax)
    @test size(state.Q)    == (solver.wmax, solver.wmax)
    @test size(state.ξ)    == (solver.wmax, 2)
    @test state.curw       == 1
    @test size(state.A)    == (solver.wmax, solver.wmax)
    @test length(state.b)  == solver.wmax
    @test length(state.xA) == length(state.x)

    # Test that tracing doesn't throw errors
    res = optimize(df, prob.initial_x, solver,
                   Optim.Options(extended_trace=true, store_trace=true;
                                 defopts...))
    @test Optim.converged(res)
    # The bounds are due to different systems behaving differently
    # TODO: is it a bad idea to hardcode these?
    @test 72 < Optim.iterations(res) < 88
    @test 245 < Optim.f_calls(res) <  292
    @test 245 < Optim.g_calls(res) <  292

    @test Optim.minimum(res) < 1e-10

    @test_throws AssertionError method(manifold=Optim.Sphere(), nlprecon = GradientDescent())

    for nlprec in (LBFGS, BFGS)
        solver = method(nlprecon=nlprec())
        clear!(df)
        res = optimize(df, prob.initial_x, solver)

        if !Optim.converged(res)
            display(res)
        end
        @test Optim.converged(res)
        @test Optim.minimum(res) < 1e-10
    end

    # O-ACCEL handles the InitialConstantChange functionality in a special way,
    # so we should test that it works well.
    for nlprec in (GradientDescent(),
                   GradientDescent(alphaguess = LineSearches.InitialConstantChange()))
        solver = method(nlprecon = nlprec,
                        alphaguess = LineSearches.InitialConstantChange())
        clear!(df)

        res = optimize(df, prob.initial_x, solver)

        if !Optim.converged(res)
            display(res)
        end
        @test Optim.converged(res)
        @test Optim.minimum(res) < 1e-10
    end
end
