@testset "Line search errors" begin
    hz = LineSearches.HagerZhang(;
        delta = 0.1,
        sigma = 0.9,
        alphamax = Inf,
        rho = 5.0,
        epsilon = 1e-6,
        gamma = 0.66,
        linesearchmax = 2,
    )
    @testset "$optimizer" for optimizer in (
        ConjugateGradient,
        GradientDescent,
        LBFGS,
        BFGS,
        Newton,
        AcceleratedGradientDescent,
        MomentumGradientDescent,
    )
        prob = MultivariateProblems.UnconstrainedProblems.examples["Exponential"]
        opt = optimize(
            MVP.objective(prob),
            prob.initial_x,
            optimizer(
                alphaguess = LineSearches.InitialStatic(; alpha = 1000.0),
                linesearch = hz,
            ),
        )
        @test opt.stopped_by.ls_failed
        @test Optim.termination_code(opt) == Optim.TerminationCode.FailedLinesearch
        @test !Optim.converged(opt)
    end

    # A line search that fails from its second call on. With a loose f_abstol and
    # successive_f_tol = 5, the first iteration sets f_converged without stopping the
    # run, and the failure in the second iteration must not be read as convergence.
    mutable struct FailFromCall{L}
        linesearch::L
        calls::Int
        fail_from::Int
    end
    function (ls::FailFromCall)(args...)
        ls.calls += 1
        if ls.calls >= ls.fail_from
            throw(LineSearches.LineSearchException("forced failure", 0.0))
        end
        return ls.linesearch(args...)
    end
    @testset "f_converged before the failure, $optimizer" for optimizer in (
        GradientDescent,
        ConjugateGradient,
        BFGS,
        LBFGS,
    )
        prob = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
        opt = optimize(
            MVP.objective(prob),
            MVP.gradient(prob),
            prob.initial_x,
            optimizer(linesearch = FailFromCall(LineSearches.HagerZhang(), 0, 2)),
            Optim.Options(f_abstol = 1e10, successive_f_tol = 5),
        )
        @test Optim.iterations(opt) == 2
        @test Optim.f_converged(opt)
        @test opt.stopped_by.ls_failed
        @test Optim.termination_code(opt) == Optim.TerminationCode.FailedLinesearch
        @test !Optim.converged(opt)
    end

    # Line searches that report success without moving x: alpha = 0, which HagerZhang
    # can return, and a step too small to change x, as HagerZhang and BackTracking
    # can take on a noisy objective
    struct FixedStep
        alpha::Float64
    end
    function (ls::FixedStep)(d, x, s, alpha, x_ls, phi_0, dphi_0)
        x_ls .= x .+ ls.alpha .* s
        return ls.alpha, phi_0
    end
    @testset "a step that does not move x, alpha = $alpha, $optimizer" for alpha in
                                                                          (0.0, 1e-300),
        optimizer in (GradientDescent, ConjugateGradient, BFGS, LBFGS, Newton)

        prob = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
        input = (MVP.objective(prob), MVP.gradient(prob))
        optimizer === Newton && (input = (input..., MVP.hessian(prob)))
        opt = optimize(input..., prob.initial_x, optimizer(linesearch = FixedStep(alpha)))
        @test Optim.iterations(opt) == 1
        @test Optim.minimizer(opt) == prob.initial_x
        @test opt.stopped_by.ls_failed
        @test Optim.termination_code(opt) == Optim.TerminationCode.FailedLinesearch
        @test !Optim.converged(opt)
    end
end
