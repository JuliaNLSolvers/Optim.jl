# TODO: add specialized tests

@testset "N-GMRES" begin
    method = NGMRES
    solver = method()

    skip = ("Trigonometric", )
    run_optim_tests(solver; skip = skip,
                    iteration_exceptions = (("Penalty Function I", 10000), ),
                    show_name = debug_printing)

    # Specialized tests
    prob = UP.examples["Rosenbrock"]
    df = OnceDifferentiable(UP.objective(prob),
                            UP.gradient(prob),
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
    # TODO: is it a bad idea to hardcode these?
    @test Optim.iterations(res) == 65
    @test Optim.f_calls(res) == 235
    @test Optim.g_calls(res) == 235
    @test Optim.minimum(res) < 1e-10

    @test_throws AssertionError method(manifold=Optim.Sphere(), nlprecon = GradientDescent())
end


@testset "O-ACCEL" begin
    method = OACCEL
    solver = method()
    skip = ("Trigonometric", )
    run_optim_tests(solver; skip = skip,
                    iteration_exceptions = (("Penalty Function I", 10000), ),
                    show_name = debug_printing)

    prob = UP.examples["Rosenbrock"]
    df = OnceDifferentiable(UP.objective(prob),
                            UP.gradient(prob),
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
    # TODO: is it a bad idea to hardcode these?
    @test Optim.iterations(res) == 87
    @test Optim.f_calls(res) == 291
    @test Optim.g_calls(res) == 291
    @test Optim.minimum(res) < 1e-10

    @test_throws AssertionError method(manifold=Optim.Sphere(), nlprecon = GradientDescent())
end
