@testset "objective types" begin
    @testset "autodiff" begin
        # Should throw, as :wah is not a proper autodiff choice
        @test_throws ErrorException OnceDifferentiable(x->x, rand(10); autodiff=:wah)

        for T in (OnceDifferentiable, TwiceDifferentiable)
            odad1 = T(x->5.0, rand(1); autodiff = :finite)
            odad2 = T(x->5.0, rand(1); autodiff = :forward)
            Optim.gradient!(odad1, rand(1))
            Optim.gradient!(odad2, rand(1))
            #    odad3 = T(x->5., rand(1); autodiff = :reverse)
            @test Optim.gradient(odad1) == [0.0]
            @test Optim.gradient(odad2) == [0.0]
            #    @test odad3.g == [0.0]
        end
        
        for a in (1.0, 5.0)
            xa = rand(1)
            odad1 = OnceDifferentiable(x->a*x[1], xa; autodiff = :finite)
            odad2 = OnceDifferentiable(x->a*x[1], xa; autodiff = :forward)
        #    odad3 = OnceDifferentiable(x->a*x[1], xa; autodiff = :reverse)        
            Optim.gradient!(odad1, xa)
            Optim.gradient!(odad2, xa)
            @test Optim.gradient(odad1) ≈ [a]
            @test Optim.gradient(odad2) == [a]
        #    @test odad3.g == [a]
        end
        for a in (1.0, 5.0)
            xa = rand(1)
            odad1 = OnceDifferentiable(x->a*x[1]^2, xa; autodiff = :finite)
            odad2 = OnceDifferentiable(x->a*x[1]^2, xa; autodiff = :forward)
        #    odad3 = OnceDifferentiable(x->a*x[1]^2, xa; autodiff = :reverse)
            Optim.gradient!(odad1, xa)
            Optim.gradient!(odad2, xa)
         @test Optim.gradient(odad1) ≈ 2.0*a*xa
            @test Optim.gradient(odad2) == 2.0*a*xa
        #    @test odad3.g == 2.0*a*xa
        end
        for dtype in (OnceDifferentiable, TwiceDifferentiable)
            for autodiff in (:finite, :forward)
                differentiable = dtype(x->sum(x), rand(2); autodiff = autodiff)
                Optim.value(differentiable)
                Optim.value!(differentiable, rand(2))
                Optim.value_gradient!(differentiable, rand(2))
                Optim.gradient!(differentiable, rand(2))
                dtype == TwiceDifferentiable && Optim.hessian!(differentiable, rand(2))
            end
        end
    end
    @testset "value/grad" begin
        a = 3.0
        x_seed = rand(1)
        odad1 = OnceDifferentiable(x->a*x[1]^2, x_seed)
        Optim.value_gradient!(odad1, x_seed)
        @test Optim.gradient(odad1) ≈ 2 .* a .* (x_seed)
        @testset "call counters" begin
            @test Optim.f_calls(odad1) == 1
            @test Optim.g_calls(odad1) == 1
            @test Optim.h_calls(odad1) == 0
            Optim.value_gradient!(odad1, x_seed .+ 1.0)
            @test Optim.f_calls(odad1) == 2
            @test Optim.g_calls(odad1) == 2
            @test Optim.h_calls(odad1) == 0
        end
        @test Optim.gradient(odad1) ≈ 2 .* a .* (x_seed .+ 1.0)
    end

end
