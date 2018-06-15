@testset "Initial Convergence Handling" begin
    f(x) = x[1]^2
    function g!(out, x)
        out[1] = 2.0 * x[1]
    end
    function h!(out, x)
        out[1,1] = 2.0
    end
    for Optimizer in (AcceleratedGradientDescent,
                      GradientDescent, ConjugateGradient, LBFGS, BFGS,
                      MomentumGradientDescent)

        res = optimize(f, g!, [0.], Optimizer())
        @test Optim.minimizer(res)[1] ≈ 0.
    end

    for Optimizer in (Newton, NewtonTrustRegion)
        res = optimize(f, g!, h!, [0.], Optimizer())
        @test Optim.minimizer(res)[1] ≈ 0.
    end
end
