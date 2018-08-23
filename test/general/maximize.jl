@testset "maximization wrapper" begin
    @testset "univariate" begin
        resmax = maximize(x->x^3, -1, 9)
        resmin = optimize(x->-x^3, -1, 9)
        @test Optim.maximum(resmax) == -Optim.minimum(resmin)
        @test resmax.res.minimum == resmin.minimum
        for meth in (Brent(), GoldenSection())
            resmax = maximize(x->x^3, -1, 9, meth)
            resmin = optimize(x->-x^3, -1, 9, meth)
            @test Optim.maximum(resmax) == -Optim.minimum(resmin)
            @test resmax.res.minimum == resmin.minimum
        end
    end
    @testset "multivariate" begin
        resmax = maximize(x->x[1]^3+x[2]^2, [3.0, 0.0])
        resmin = optimize(x->-x[1]^3-x[2]^2, [3.0, 0.0])
        @test Optim.maximum(resmax) == -Optim.minimum(resmin)
        @test resmax.res.minimum == resmin.minimum
        for meth in (NelderMead(), BFGS(), LBFGS(), GradientDescent(), Newton(), NewtonTrustRegion(), SimulatedAnnealing())
            resmax = maximize(x->x[1]^3+x[2]^2, [3.0, 0.0])
            resmin = optimize(x->-x[1]^3-x[2]^2, [3.0, 0.0])
            @test Optim.maximum(resmax) == -Optim.minimum(resmin)
            @test resmax.res.minimum == resmin.minimum
        end
    end
end
