@testset "IPNewton Unconstrained" begin
    method = IPNewton()
    run_optim_tests_constrained(method; show_name=debug_printing, show_res=false, show_itcalls=debug_printing)

    # prob = UP.examples["Rosenbrock"]
    # df = TwiceDifferentiable(UP.objective(prob), UP.gradient(prob),
    #                          UP.objective_gradient(prob), UP.hessian(prob), prob.initial_x)

    # infvec = fill(Inf, size(prob.initial_x))
    # constraints = TwiceDifferentiableConstraints(-infvec, infvec)

    # options = Options(; ConstrainedOptim.default_options(method)...)

    # results = optimize(df,constraints,prob.initial_x, method, options)
    # @test isa(summary(results), String)
    # @test ConstrainedOptim.converged(results)
    # @test ConstrainedOptim.minimum(results) < prob.minimum + sqrt(eps(typeof(prob.minimum)))
    # @test vecnorm(ConstrainedOptim.minimizer(results) - prob.solutions) < 1e-2
end
