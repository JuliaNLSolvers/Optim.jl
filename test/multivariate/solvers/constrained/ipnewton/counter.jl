@testset "Function counter" begin
    prob = OptimTestProblems.UnconstrainedProblems.examples["Rosenbrock"]

    let
        global fcount = 0
        global fcounter
        function fcounter(reset::Bool = false)
            if reset
                fcount = 0
            else
                fcount += 1
            end
            fcount
        end
        global gcount = 0
        global gcounter
        function gcounter(reset::Bool = false)
            if reset
                gcount = 0
            else
                gcount += 1
            end
            gcount
        end
        global hcount = 0
        global hcounter
        function hcounter(reset::Bool = false)
            if reset
                hcount = 0
            else
                hcount += 1
            end
            hcount
        end
    end

    f(x) = begin
        fcounter()
        MVP.objective(prob)(x)
    end
    g!(out, x) = begin
        gcounter()
        MVP.gradient(prob)(out, x)
    end
    h!(out, x) = begin
        hcounter()
        MVP.hessian(prob)(out, x)
    end

    options = Optim.Options(; Optim.default_options(IPNewton())...)
    # TODO: Run this on backtrack_constrained as well (when we figure out what it does)
    for ls in [Optim.backtrack_constrained_grad,]
               #Optim.backtrack_constrained]

        fcounter(true); gcounter(true); hcounter(true)

        df = TwiceDifferentiable(f, g!, h!, prob.initial_x)
        infvec = fill(Inf, size(prob.initial_x))
        # TODO: fix `optimize`so that empty constraints work
        #constraints = TwiceDifferentiableConstraints([], [])
        constraints = TwiceDifferentiableConstraints(-infvec, infvec)

        res = optimize(df, constraints, prob.initial_x,
                       IPNewton(linesearch = ls),
                       options)
        @test fcount == Optim.f_calls(res)
        @test gcount == Optim.g_calls(res)
        @test hcount == Optim.h_calls(res)
    end

    # TODO: Test a constrained problem

    # TODO: Test constraints call counter (if we need to make one)
end
