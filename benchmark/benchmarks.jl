using BenchmarkTools
using Optim
using Random: MersenneTwister, seed!

const SUITE = BenchmarkGroup()

# Example function to optimize
function gabor(x, phi)
    # Optimization example from "Understanding Deep Learning"; Prince, 2023
    return sin(phi[1] + 0.06 * phi[2] * x) * exp(-(phi[1] + 0.06 * phi[1] * x)^2 / 32.0)
end

gabor_problem = (;
    loss_generator = (X, Y) ->
        (phi -> sum(i -> (gabor(X[i], phi) - Y[i])^2, eachindex(X, Y))),
    init_phi = () -> [1.0, 6.0],
    true_phi = () -> [0.0, 16.6],
    domain = () -> LinRange(-15.0, 15.0, 64),
    options = () -> Optim.Options(iterations = 100),
)

tests = (;
    first_order = (;
        gabor_problem...,
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
    ),
    second_order = (;
        gabor_problem...,
        optimizers = [:Newton, :NewtonTrustRegion, :KrylovTrustRegion],
    ),
)

for order in keys(tests), optimizer in tests[order].optimizers
    isdefined(Optim, optimizer) || continue
    SUITE["multivariate"]["solvers"][order][optimizer] = @benchmarkable(
        optimize(loss, init_phi, opt, options),
        setup = (test = $(tests[order]);
        init_phi = test.init_phi();
        true_phi = test.true_phi();
        opt = $(getproperty(Optim, optimizer))();
        options = test.options();
        rng = MersenneTwister(0);
        X = collect(test.domain());
        noise = 0.1 * randn(rng, length(X));
        Y = map(i -> gabor(X[i], true_phi) + noise[i], eachindex(X, noise));
        loss = test.loss_generator(X, Y);
        seed!(1))
    )
end

# A problem that can be grown, to expose how the solvers scale with the number of
# variables. Its Hessian is indefinite at the starting point, so the trust region
# solvers spend their first iterations on the boundary of the trust region.
rosenbrock(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i = 1:2:length(x))

function rosenbrock_gradient!(G, x)
    for i = 1:2:length(x)
        G[i] = -400 * x[i] * (x[i+1] - x[i]^2) - 2 * (1 - x[i])
        G[i+1] = 200 * (x[i+1] - x[i]^2)
    end
    return G
end

function rosenbrock_hessian!(H, x)
    fill!(H, 0)
    for i = 1:2:length(x)
        H[i, i] = 1200 * x[i]^2 - 400 * x[i+1] + 2
        H[i, i+1] = -400 * x[i]
        H[i+1, i] = -400 * x[i]
        H[i+1, i+1] = 200
    end
    return H
end

# A separable saddle, whose Hessian is indefinite over a wide region rather than only
# at the starting point. The trust region solvers work on the boundary throughout and
# reject steps as the radius adapts, around 40% of them here, so the cell is sensitive
# both to how well the subproblem is solved and to what a rejected step costs. On
# Rosenbrock, by contrast, under a fifth of the steps are rejected.
saddle(x) = sum(x[i]^4 - 8 * x[i]^2 + 3 * x[i] * x[i+1] + x[i+1]^2 for i = 1:2:length(x))

function saddle_gradient!(G, x)
    for i = 1:2:length(x)
        G[i] = 4 * x[i]^3 - 16 * x[i] + 3 * x[i+1]
        G[i+1] = 3 * x[i] + 2 * x[i+1]
    end
    return G
end

function saddle_hessian!(H, x)
    fill!(H, 0)
    for i = 1:2:length(x)
        H[i, i] = 12 * x[i]^2 - 16
        H[i, i+1] = 3
        H[i+1, i] = 3
        H[i+1, i+1] = 2
    end
    return H
end

# Derivatives are supplied, and the iteration budget is generous, so that the timings
# measure the solvers rather than finite differences or a truncated run
for n in (2, 20, 100), optimizer in (:Newton, :NewtonTrustRegion, :KrylovTrustRegion)
    isdefined(Optim, optimizer) || continue
    SUITE["multivariate"]["problems"]["rosenbrock"][n][optimizer] = @benchmarkable(
        optimize(rosenbrock, rosenbrock_gradient!, rosenbrock_hessian!, x0, opt, options),
        setup = (x0 = repeat([-1.2, 1.0], $n ÷ 2);
        opt = $(getproperty(Optim, optimizer))();
        options = Optim.Options(iterations = 1_000))
    )
    SUITE["multivariate"]["problems"]["saddle"][n][optimizer] = @benchmarkable(
        optimize(saddle, saddle_gradient!, saddle_hessian!, x0, opt, options),
        setup = (x0 = repeat([0.1, -0.1], $n ÷ 2);
        opt = $(getproperty(Optim, optimizer))();
        options = Optim.Options(iterations = 1_000))
    )
end
