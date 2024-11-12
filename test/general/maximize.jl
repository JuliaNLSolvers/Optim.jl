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

    resmax_f = maximize(fmax, prob.initial_x)
    resmin_f = optimize(f, prob.initial_x)
    for prop in (:iterations, :ls_success, :minimizer, :minimum)
        @test getproperty(resmax_f.res, prop) == getproperty(resmin_f, prop)
    end

    resmax_f_nm = maximize(fmax, prob.initial_x, NelderMead())
    resmin_f_nm = optimize(f, prob.initial_x, NelderMead())
    for prop in (:iterations, :ls_success, :minimizer, :minimum)
        @test getproperty(resmax_f_nm.res, prop) == getproperty(resmin_f_nm, prop)
    end

    resmax_f_bfgs = maximize(fmax, prob.initial_x, BFGS())
    resmin_f_bfgs = optimize(f, prob.initial_x, BFGS())
    for prop in (:iterations, :ls_success, :minimizer, :minimum)
        @test getproperty(resmax_f_bfgs.res, prop) == getproperty(resmin_f_bfgs, prop)
    end

    resmax_f_newton = maximize(fmax, prob.initial_x, Newton())
    resmin_f_newton = optimize(f, prob.initial_x, Newton())
    for prop in (:iterations, :ls_success, :minimizer, :minimum)
        @test getproperty(resmax_f_newton.res, prop) == getproperty(resmin_f_newton, prop)
    end

    resmax_fg = maximize(fmax, gmax, prob.initial_x, BFGS())
    resmin_fg = optimize(f, g!, prob.initial_x, BFGS())
    for prop in (:iterations, :ls_success, :minimizer, :minimum)
        @test getproperty(resmax_fg.res, prop) == getproperty(resmin_fg, prop)
    end

    resmax_fgh = maximize(fmax, gmax, hmax, prob.initial_x, Newton())
    resmin_fgh = optimize(f, g!, h!, prob.initial_x, Newton())
    for prop in (:iterations, :ls_success, :minimizer, :minimum)
        @test getproperty(resmax_fgh.res, prop) == getproperty(resmin_fgh, prop)
    end

end
