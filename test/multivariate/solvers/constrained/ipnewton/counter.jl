@testset "Function counter" begin
    options = Optim.Options(; Optim.default_options(IPNewton())...)
    # TODO: Run this on backtrack_constrained as well (when we figure out what it does)
    for ls in [Optim.backtrack_constrained_grad]
        #Optim.backtrack_constrained]

        prob, f, g!, h!, calls = counted_rosenbrock()
        df = TwiceDifferentiable(f, g!, h!, prob.initial_x)
        infvec = fill(Inf, size(prob.initial_x))
        # TODO: fix `optimize`so that empty constraints work
        #constraints = TwiceDifferentiableConstraints([], [])
        constraints = TwiceDifferentiableConstraints(-infvec, infvec)

        res = optimize(df, constraints, prob.initial_x, IPNewton(linesearch = ls), options)
        @test calls.f[] == Optim.f_calls(res)
        @test calls.g[] == Optim.g_calls(res)
        @test calls.h[] == Optim.h_calls(res)
    end

    # TODO: Test a constrained problem

    # TODO: Test constraints call counter (if we need to make one)
end
