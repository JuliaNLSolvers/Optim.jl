using Optim, Test, Random
@testset "#711" begin
    # make sure it doesn't try to promote df
    dof = 7

    fun(x) = 0.0
    x0 = fill(0.1, dof)
    df = TwiceDifferentiable(fun, x0)

    lx = fill(-1.2, dof)
    ux = fill(+1.2, dof)
    dfc = TwiceDifferentiableConstraints(lx, ux)

    res = optimize(df, dfc, x0, IPNewton(); autodiff = AutoForwardDiff())
    res = optimize(df, dfc, x0, IPNewton())
end

@testset "#600" begin
    function exponential(x)
        return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
    end

    function exponential_gradient!(storage, x)
        storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
        storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
        storage
    end
    function exponential_hessian!(storage, x)
        ForwardDiff.hessian!(storage, exponential, x)
    end

    function exponential_gradient(x)
        storage = similar(x)
        storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
        storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
        storage
    end

    initial_x = [0.0, 0.0]
    optimize(exponential, exponential_gradient!, initial_x, BFGS())
    lb = fill(-0.1, 2)
    ub = fill(1.1, 2)
    od = OnceDifferentiable(exponential, initial_x)
    optimize(od, lb, ub, initial_x, IPNewton())
    optimize(od, lb, ub, initial_x, IPNewton(), Optim.Options())
    optimize(exponential, lb, ub, initial_x, IPNewton())
    optimize(exponential, lb, ub, initial_x, IPNewton(), Optim.Options())
    optimize(exponential, exponential_gradient!, lb, ub, initial_x, IPNewton())
    optimize(
        exponential,
        exponential_gradient!,
        lb,
        ub,
        initial_x,
        IPNewton(),
        Optim.Options(),
    )
    optimize(
        exponential,
        exponential_gradient!,
        exponential_hessian!,
        lb,
        ub,
        initial_x,
        IPNewton(),
    )
    optimize(
        exponential,
        exponential_gradient!,
        exponential_hessian!,
        lb,
        ub,
        initial_x,
        IPNewton(),
        Optim.Options(),
    )
    optimize(TwiceDifferentiable(od, initial_x), lb, ub, initial_x)
    optimize(TwiceDifferentiable(od, initial_x), lb, ub, initial_x, Optim.Options())
end

@testset "non-terminating line search (αmax -> 0)" begin
    # Density-ratio (KLIEP) problem that used to hang IPNewton: as the barrier
    # parameter μ collapses, a slack/multiplier underflows to 0, `estimate_maxstep`
    # returns `-0/negative == 0`, so the line search saw `α == αmin == 0` and the
    # `while α >= αmin` loop never terminated (`0 >= 0` stays true, `α *= ρ` keeps
    # α at 0). The guard `α > 0` makes the search take a zero step and return.
    Random.seed!(1234)
    σ = 2.0
    b = 10
    euclidsq(x, y) = sum((x[i] - y[i])^2 for i in eachindex(x))
    gramian(xs, ys; σ = 1) = [exp(-euclidsq(x, y) / 2σ^2) for x in xs, y in ys]

    x_nu, x_de = 5randn(100), randn(100)
    x_ba = x_nu[1:b]
    K = gramian(x_nu, x_ba, σ = σ)        # numerator gram matrix, used in objective
    K_de = gramian(x_de, x_ba, σ = σ)
    n_nu, n_de = size(K, 1), size(K_de, 1)

    A = sum(K_de, dims = 1)
    lc = uc = [float(n_de)]
    lx = fill(0.0, b)
    ux = fill(Inf, b)

    f(α) = -sum(log, K * α)
    function ∇f!(g, α)
        p = K * α
        for l = 1:b
            g[l] = -sum(K[j, l] / p[j] for j = 1:n_nu)
        end
        g
    end
    function ∇²f!(h, α)
        p = K * α
        for k = 1:b, l = 1:b
            h[k, l] = sum(view(K, :, k) .* view(K, :, l) ./ p)
        end
        h
    end
    c!(c, α) = c .= A * α
    J!(J, α) = J .= A
    H!(H, α, λ) = H .+= 0

    α₀ = fill(n_de / sum(A), b)
    obj = TwiceDifferentiable(f, ∇f!, ∇²f!, α₀)
    con = TwiceDifferentiableConstraints(c!, J!, H!, lx, ux, lc, uc)

    # The real regression check is that this call returns at all (no hang).
    local res
    elapsed = @elapsed(res = optimize(obj, con, α₀, IPNewton()))
    @test Optim.iterations(res) <= 1000          # terminated within the default iteration cap
    @test isfinite(Optim.minimum(res))
    @test all(isfinite, Optim.minimizer(res))
    # With the fix this converges in well under a second; the previous behaviour
    # was a hang of many minutes. A generous bound still flags a regression.
    @test elapsed < 60
end
