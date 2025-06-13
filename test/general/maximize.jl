@testset "maximization wrapper" begin
    @testset "univariate" begin
        resmax = maximize(x -> x^3, -1, 9)
        resmin = optimize(x -> -x^3, -1, 9)
        @test Optim.maximum(resmax) == -Optim.minimum(resmin)
        @test resmax.res.minimum == resmin.minimum
        for meth in (Brent(), GoldenSection())
            resmax = maximize(x -> x^3, -1, 9, meth)
            resmin = optimize(x -> -x^3, -1, 9, meth)
            @test Optim.maximum(resmax) == -Optim.minimum(resmin)
            @test resmax.res.minimum == resmin.minimum
        end
    end
    @testset "multivariate" begin
        resmax = maximize(x -> x[1]^3 + x[2]^2, [3.0, 0.0])
        resmin = optimize(x -> -x[1]^3 - x[2]^2, [3.0, 0.0])
        @test Optim.maximum(resmax) == -Optim.minimum(resmin)
        @test resmax.res.minimum == resmin.minimum
        for meth in (
            NelderMead(),
            BFGS(),
            LBFGS(),
            GradientDescent(),
            Newton(),
            NewtonTrustRegion(),
            SimulatedAnnealing(),
        )
            resmax = maximize(x -> x[1]^3 + x[2]^2, [3.0, 0.0])
            resmin = optimize(x -> -x[1]^3 - x[2]^2, [3.0, 0.0])
            @test Optim.maximum(resmax) == -Optim.minimum(resmin)
            @test resmax.res.minimum == resmin.minimum
        end
    end

    prob = MVP.UnconstrainedProblems.examples["Powell"]
    f = objective(prob)
    g! = gradient(prob)
    h! = hessian(prob)
    fmax(x) = -f(x)
    gmax = (G, x) -> (g!(G, x); G .= -G)
    hmax = (H, x) -> (h!(H, x); H .= -H)

    function test_same_content(f1, f2)
        for prop in (:minimizer, :minimum)
            @test getproperty(f1.res, prop) == getproperty(f2, prop)
        end
        for prop in (:iterations, :ls_failed)
            @test getproperty(f1.res.stopped_by, prop) == getproperty(f2.stopped_by, prop)
        end
    end

    resmax_f = maximize(fmax, prob.initial_x)
    resmin_f = optimize(f, prob.initial_x)

    for alg in (NelderMead(), BFGS(), Newton())
        resmax_f = maximize(fmax, prob.initial_x, alg)
        resmin_f = optimize(f, prob.initial_x, alg)
        test_same_content(resmax_f, resmin_f)
    end

    for alg in (NelderMead(), BFGS(), Newton())
        resmax_fg = maximize(fmax, gmax, prob.initial_x, BFGS())
        resmin_fg = optimize(f, g!, prob.initial_x, BFGS())
        test_same_content(resmax_fg, resmin_fg)
    end

    resmax_fgh = maximize(fmax, gmax, hmax, prob.initial_x, Newton())
    resmin_fgh = optimize(f, g!, h!, prob.initial_x, Newton())
    test_same_content(resmax_fgh, resmin_fgh)
end
