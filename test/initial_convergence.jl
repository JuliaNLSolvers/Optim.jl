let
    f(x) = x[1]^2
    function g!(x, out)
        out[1] = 2.*x[1]
    end
    function h!(x, out)
        out[1,1] = 2.
    end
    for Optimizer in (AcceleratedGradientDescent,
                      GradientDescent, ConjugateGradient, LBFGS, BFGS,
                      MomentumGradientDescent)

        res = optimize(f, g!, [0.], Optimizer())
        @assert isapprox(res.minimum[1], 0.)
    end

    for Optimizer in (Newton, NewtonTrustRegion)
        res = optimize(f, g!, h!, [0.], Optimizer())
        @assert isapprox(res.minimum[1], 0.)
    end
end
