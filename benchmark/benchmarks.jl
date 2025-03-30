using BenchmarkTools
using Optim
using Random: MersenneTwister, seed!

const SUITE = BenchmarkGroup()

# Example function to optimize
function gabor(x, phi)
    # Optimization example from "Understanding Deep Learning"; Prince, 2023
    return sin(phi[1] + 0.06 * phi[2] * x) * exp(-(phi[1] + 0.06 * phi[1] * x)^2 / 32.0)
end

tests = (;
    first_order = (;
        loss_generator = (X, Y) ->
            (phi -> sum(i -> (gabor(X[i], phi) - Y[i])^2, eachindex(X, Y))),
        init_phi = () -> [1.0, 6.0],
        true_phi = () -> [0.0, 16.6],
        domain = () -> LinRange(-15.0, 15.0, 64),
        options = () -> Optim.Options(iterations = 100),
        optimizers = [
            :Adam,
            :AdaMax,
            :BFGS,
            :LBFGS,
            :NGMRES,
            :ConjugateGradient,
            :GradientDescent,
            :MomentumGradientDescent,
        ],
    )
)

for order in keys(tests), optimizer in tests[order].optimizers
    isdefined(@__MODULE__, optimizer) || continue
    SUITE["multivariate"]["solvers"][order][optimizer] = @benchmarkable(
        optimize(loss, init_phi, opt, options),
        setup = (test = $(tests[order]);
        init_phi = test.init_phi();
        true_phi = test.true_phi();
        opt = $(eval(optimizer))();
        options = test.options();
        rng = MersenneTwister(0);
        X = collect(test.domain());
        noise = 0.1 * randn(rng, length(X));
        Y = map(i -> gabor(X[i], true_phi) + noise[i], eachindex(X, noise));
        loss = test.loss_generator(X, Y);
        seed!(1))
    )
end


results = run(SUITE)
