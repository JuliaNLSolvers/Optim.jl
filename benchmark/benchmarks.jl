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

# The gradient is O(n), so for larger n the timings are dominated by the O(n^2) update
# of the inverse Hessian approximation
for n in (2, 20, 100), optimizer in (:BFGS,)
    SUITE["multivariate"]["problems"]["rosenbrock"][n][optimizer] = @benchmarkable(
        optimize(rosenbrock, rosenbrock_gradient!, x0, opt, options),
        setup = (x0 = repeat([-1.2, 1.0], $n ÷ 2);
        opt = $(getproperty(Optim, optimizer))();
        options = Optim.Options(iterations = 1_000))
    )
end

# The saddle's minimum is about -13.1n, and near it the predicted reduction falls below
# one ulp of f, so roundoff alone makes the objective noisy. Without a noise level the
# radius collapses there and the run ends on the f-change test above g_abstol, so the
# cells above also run NewtonTrustRegion with that rounding level as its noise floor,
# where the version has the keyword.
if :epsilon_f in fieldnames(NewtonTrustRegion)
    for n in (2, 20, 100)
        SUITE["multivariate"]["problems"]["saddle"][n][:NewtonTrustRegion_noise_floor] = @benchmarkable(
            optimize(saddle, saddle_gradient!, saddle_hessian!, x0, opt, options),
            setup = (x0 = repeat([0.1, -0.1], $n ÷ 2);
            opt = NewtonTrustRegion(epsilon_f = eps(13.2 * $n));
            options = Optim.Options(iterations = 1_000))
        )
    end
end

# A quadratic with noise on the objective and the gradient, like an objective that is
# itself computed by an iterative method to a fixed tolerance. The noise is the hash
# of x, so it is a fixed function of x and master and the pull request see the same
# problem; Base.hash can change between Julia versions, which only matters when
# comparing numbers across versions. g_abstol sits below the gradient noise, so the
# runs end where the line search can no longer move x, which exercises the check for
# that in perform_linesearch! and the retry it triggers. Where that point lands
# depends on rounding, so a change that reorders floating point work elsewhere can
# move these timings; the allocation counts show whether the amount of work changed.
noise(x, salt) = 2 * ((hash(x, salt) >> 11) * 2.0^-53) - 1

const NOISY_N = 8
const NOISY_A = [1 / (i + j - 1) + (i == j ? NOISY_N : 0) for i = 1:NOISY_N, j = 1:NOISY_N]
const NOISY_XSTAR = collect(range(0.5, 1.5, length = NOISY_N))

function noisy_quadratic(x)
    r = x .- NOISY_XSTAR
    return sum(r[i] * NOISY_A[i, j] * r[j] for i = 1:NOISY_N, j = 1:NOISY_N) / 2 +
           1e-8 * noise(x, UInt(1))
end

function noisy_quadratic_gradient!(G, x)
    r = x .- NOISY_XSTAR
    for i = 1:NOISY_N
        G[i] =
            sum(NOISY_A[i, j] * r[j] for j = 1:NOISY_N) +
            1e-9 * noise(x, UInt(1 + i))
    end
    return G
end

for optimizer in (:BFGS, :LBFGS, :ConjugateGradient),
    linesearch in (:HagerZhang, :BackTracking)

    isdefined(Optim, optimizer) || continue
    SUITE["multivariate"]["problems"]["noisy_quadratic"][NOISY_N][optimizer][linesearch] =
        @benchmarkable(
            optimize(noisy_quadratic, noisy_quadratic_gradient!, x0, opt, options),
            setup = (x0 = zeros(NOISY_N);
            opt = $(getproperty(Optim, optimizer))(
                linesearch = $(getproperty(Optim.LineSearches, linesearch))(),
            );
            options = Optim.Options(g_abstol = 1e-10, iterations = 5_000))
        )
end

# A convex quadratic x'Ax/2 - b'x with A = tridiag(-1, 2.01, -1), condition number
# about 400, for LBFGSB. b is chosen so that the minimizer in [-1, 1]^n has a random
# half of the variables at a bound, on random sides, so the pattern of active
# variables and gradient signs is not one a branch predictor can learn. The same
# quadratic without bounds exercises the path where every variable is free.
tridiagonal_product(x, i) =
    2.01 * x[i] - (i > 1 ? x[i-1] : zero(eltype(x))) -
    (i < length(x) ? x[i+1] : zero(eltype(x)))

function tridiagonal_rhs(n)
    rng = MersenneTwister(0)
    side = [rand(rng) < 0.5 ? 0 : rand(rng, (-1, 1)) for _ = 1:n]
    xstar = [s == 0 ? 2 * rand(rng) - 1 : float(s) for s in side]
    # the gradient at xstar: zero on the free variables, and pointing into the box on
    # the active ones, so that -g points out of it
    gstar = [-s * rand(rng) for s in side]
    return [tridiagonal_product(xstar, i) - gstar[i] for i = 1:n]
end

tridiagonal_quadratic(x, b) =
    sum(i -> x[i] * (tridiagonal_product(x, i) / 2 - b[i]), eachindex(x, b))

function tridiagonal_quadratic_gradient!(G, x, b)
    for i in eachindex(G, x, b)
        G[i] = tridiagonal_product(x, i) - b[i]
    end
    return G
end

if isdefined(Optim, :LBFGSB)
    for n in (100, 10_000), (bounds, lu) in (("box", (-1.0, 1.0)), ("none", (-Inf, Inf)))
        SUITE["multivariate"]["problems"]["tridiagonal_quadratic"][n][bounds][:LBFGSB] =
            @benchmarkable(
                optimize(
                    x -> tridiagonal_quadratic(x, b),
                    (G, x) -> tridiagonal_quadratic_gradient!(G, x, b),
                    l,
                    u,
                    x0,
                    LBFGSB(),
                    Optim.Options(),
                ),
                setup = (b = tridiagonal_rhs($n);
                l = fill($(lu[1]), $n);
                u = fill($(lu[2]), $n);
                x0 = zeros($n))
            )
    end
end
