@testset "objective types" begin
    @testset "autodiff" begin
        for T in (OnceDifferentiable, TwiceDifferentiable)
            odad1 = T(x -> 5.0, rand(1); autodiff = AutoFiniteDiff(; fdtype = Val(:central)))
            odad2 = T(x -> 5.0, rand(1); autodiff = AutoForwardDiff())
            odad3 = T(x -> 5.0, rand(1); autodiff = AutoReverseDiff())
            NLSolversBase.gradient!(odad1, rand(1))
            NLSolversBase.gradient!(odad2, rand(1))
            NLSolversBase.gradient!(odad3, rand(1))
            @test NLSolversBase.gradient(odad1) == [0.0]
            @test NLSolversBase.gradient(odad2) == [0.0]
            @test NLSolversBase.gradient(odad3) == [0.0]
        end

        for a in (1.0, 5.0)
            xa = rand(1)
            odad1 = OnceDifferentiable(x -> a * x[1], xa; autodiff = AutoFiniteDiff(; fdtype = Val(:central)))
            odad2 = OnceDifferentiable(x -> a * x[1], xa; autodiff = AutoForwardDiff())
            odad3 = OnceDifferentiable(x -> a * x[1], xa; autodiff = AutoReverseDiff())
            NLSolversBase.gradient!(odad1, xa)
            NLSolversBase.gradient!(odad2, xa)
            NLSolversBase.gradient!(odad3, xa)
            @test NLSolversBase.gradient(odad1) ≈ [a]
            @test NLSolversBase.gradient(odad2) == [a]
            @test NLSolversBase.gradient(odad3) == [a]
        end
        for a in (1.0, 5.0)
            xa = rand(1)
            odad1 = OnceDifferentiable(x -> a * x[1]^2, xa; autodiff = AutoFiniteDiff(; fdtype = Val(:central)))
            odad2 = OnceDifferentiable(x -> a * x[1]^2, xa; autodiff = AutoForwardDiff())
            odad3 = OnceDifferentiable(x -> a * x[1]^2, xa; autodiff = AutoReverseDiff())
            NLSolversBase.gradient!(odad1, xa)
            NLSolversBase.gradient!(odad2, xa)
            NLSolversBase.gradient!(odad3, xa)
            @test NLSolversBase.gradient(odad1) ≈ 2.0 * a * xa
            @test NLSolversBase.gradient(odad2) == 2.0 * a * xa
            @test NLSolversBase.gradient(odad3) == 2.0 * a * xa
        end
        for dtype in (OnceDifferentiable, TwiceDifferentiable)
            for autodiff in (AutoFiniteDiff(; fdtype = Val(:central)), AutoForwardDiff(), AutoReverseDiff())
                differentiable = dtype(x -> sum(x), rand(2); autodiff = autodiff)
                NLSolversBase.value(differentiable)
                NLSolversBase.value!(differentiable, rand(2))
                NLSolversBase.value_gradient!(differentiable, rand(2))
                NLSolversBase.gradient!(differentiable, rand(2))
                dtype == TwiceDifferentiable && NLSolversBase.hessian!(differentiable, rand(2))
            end
        end
    end
    @testset "value/grad" begin
        a = 3.0
        x_seed = rand(1)
        odad1 = OnceDifferentiable(x -> a * x[1]^2, x_seed)
        NLSolversBase.value_gradient!(odad1, x_seed)
        @test NLSolversBase.gradient(odad1) ≈ 2 .* a .* (x_seed)
        @testset "call counters" begin
            @test NLSolversBase.f_calls(odad1) == 1
            @test NLSolversBase.g_calls(odad1) == 1
            @test NLSolversBase.h_calls(odad1) == 0
            NLSolversBase.value_gradient!(odad1, x_seed .+ 1.0)
            @test NLSolversBase.f_calls(odad1) == 2
            @test NLSolversBase.g_calls(odad1) == 2
            @test NLSolversBase.h_calls(odad1) == 0
        end
        @test NLSolversBase.gradient(odad1) ≈ 2 .* a .* (x_seed .+ 1.0)
    end

end
