@testset "function counter" begin
    ls = LineSearches.HagerZhang()

    for solver in (
        AcceleratedGradientDescent,
        BFGS,
        ConjugateGradient,
        GradientDescent,
        LBFGS,
        MomentumGradientDescent,
        NGMRES,
        OACCEL,
    )
        prob, f, g!, h!, calls = counted_rosenbrock()
        res = Optim.optimize(f, g!, prob.initial_x, solver(linesearch = ls))
        @test calls.f[] == Optim.f_calls(res)
        @test calls.g[] == Optim.g_calls(res)
    end

    for solver in (Newton(linesearch = ls), NewtonTrustRegion())
        prob, f, g!, h!, calls = counted_rosenbrock()
        res = Optim.optimize(f, g!, h!, prob.initial_x, solver)
        @test calls.f[] == Optim.f_calls(res)
        @test calls.g[] == Optim.g_calls(res)
        @test calls.h[] == Optim.h_calls(res)
    end

    @testset "KrylovTrustRegion" begin
        prob, f, g!, h!, calls = counted_rosenbrock()
        fg!(F, G, x) = begin
            if G !== nothing
                g!(G, x)
            end
            return F === nothing ? nothing : f(x)
        end
        _hvp!(out, x, v) = begin
            n = length(x)
            H = Matrix{Float64}(undef, n, n)
            h!(H, x)
            out .= H * v
        end
        df = Optim.TwiceDifferentiable(NLSolversBase.only_fg_and_hvp!(fg!, _hvp!), prob.initial_x)
        res = Optim.optimize(df, prob.initial_x, Optim.KrylovTrustRegion())
        @test calls.f[] == Optim.f_calls(res)
        @test calls.g[] == Optim.g_calls(res)
        @test calls.h[] == Optim.hvp_calls(res)
    end
end
