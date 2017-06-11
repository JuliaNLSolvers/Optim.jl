@testset "objective types" begin
    @testset "autodiff" begin
        # Should throw, as :wah is not a proper autodiff choice
        @test_throws ErrorException OnceDifferentiable(x->x, rand(10); autodiff=:wah)

        for T in (OnceDifferentiable, TwiceDifferentiable)
            odad1 = T(x->5., rand(1); autodiff = :finite)
            odad2 = T(x->5., rand(1); autodiff = :forward)
        #    odad3 = T(x->5., rand(1); autodiff = :reverse)
            @test odad1.g == [0.0]
            @test odad2.g == [0.0]
        #    @test odad3.g == [0.0]
        end

        for a in (1.0, 5.0)
            odad1 = OnceDifferentiable(x->a*x[1], rand(1); autodiff = :finite)
            odad2 = OnceDifferentiable(x->a*x[1], rand(1); autodiff = :forward)
        #    odad3 = OnceDifferentiable(x->a*x[1], rand(1); autodiff = :reverse)
            @test odad1.g ≈ [a]
            @test odad2.g == [a]
        #    @test odad3.g == [a]
        end
        for a in (1.0, 5.0)
            x_seed = rand(1)
            odad1 = OnceDifferentiable(x->a*x[1]^2, x_seed; autodiff = :finite)
            odad2 = OnceDifferentiable(x->a*x[1]^2, x_seed; autodiff = :forward)
        #    odad3 = OnceDifferentiable(x->a*x[1]^2, x_seed; autodiff = :reverse)
            @test odad1.g ≈ 2.0*a*x_seed
            @test odad2.g == 2.0*a*x_seed
        #    @test odad3.g == 2.0*a*x_seed
        end
        for dtype in (OnceDifferentiable, TwiceDifferentiable)
            for autodiff in (:finite, :forward)
                differentiable = OnceDifferentiable(x->sum(x), rand(2); autodiff = autodiff)
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
        odad1 = OnceDifferentiable(x->a*x[1]^2, x_seed)
        tmp_1 = copy(odad1.g)
        Optim.value_gradient!(odad1, x_seed)
        @test tmp_1 == odad1.g
        @testset "call counters" begin
            @test Optim.f_calls(odad1) == 1
            @test Optim.g_calls(odad1) == 1
            @test Optim.h_calls(odad1) == 0
            Optim.value_gradient!(odad1, x_seed+1.0)
            @test Optim.f_calls(odad1) == 2
            @test Optim.g_calls(odad1) == 2
            @test Optim.h_calls(odad1) == 0
        end
        @test Optim.gradient(odad1) ≈ 2*a*(x_seed+1.0)
    end

end
