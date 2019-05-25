@testset "bigfloat initial convergence #720" begin
    f(x) = x[1]^2
    x0 = BigFloat[0]
    obj = OnceDifferentiable(f, x0; autodiff=:forward)
    for method in (GradientDescent, BFGS, LBFGS, AcceleratedGradientDescent, MomentumGradientDescent, ConjugateGradient, Newton, NewtonTrustRegion, SimulatedAnnealing)
        result = Optim.optimize(f, x0, method())
    end
end
