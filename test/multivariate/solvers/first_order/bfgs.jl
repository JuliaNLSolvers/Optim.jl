using Optim, Test
@testset "BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",)
    run_optim_tests(BFGS(); convergence_exceptions = (("Polynomial",1),),
                    f_increase_exceptions = ("Extended Rosenbrock",),
                    skip=skip,
                    show_name = debug_printing)
end
@testset "reset" begin
    rosenbrock = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
    f = MVP.objective(rosenbrock)
    g! = MVP.gradient(rosenbrock)
    h! = MVP.hessian(rosenbrock)

    initial_x = rosenbrock.initial_x
    
    d = OnceDifferentiable(f, g!, initial_x)

    options= Optim.Options()
    state = Optim.initial_state(method, options, d, initial_x)
    initial_invH = [ 0.0004638218923933211, 0.0004638218923933211]
    @test all(state.invH .== initial_invH)
    op_res  = optimize(d, initial_x, method, options, state)
end
