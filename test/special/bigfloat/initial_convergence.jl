@testset "bigfloat initial convergence #720" begin
    f(x) = x[1]^2
    x0 = BigFloat[0]
    for method in (
        GradientDescent,
        BFGS,
        LBFGS,
        AcceleratedGradientDescent,
        MomentumGradientDescent,
        ConjugateGradient,
        Newton,
        NewtonTrustRegion,
    )
        result = Optim.optimize(f, x0, method())
        @test Optim.iterations(result) == 0
        @test Optim.converged(result)
        @test eltype(Optim.minimizer(result)) == BigFloat
    end
    # SimulatedAnnealing has no initial convergence check and runs to the iteration limit
    result = Optim.optimize(f, x0, SimulatedAnnealing())
    @test eltype(Optim.minimizer(result)) == BigFloat
end
