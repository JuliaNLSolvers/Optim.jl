# Simple reference L-BFGS-B, kept internal (NOT exported) purely so the
# efficient `LBFGSB` solver in lbfgsb.jl can be compared against it. This
# Was my first implementation without all the optimizations
#
# Same algorithm as `LBFGSB` (generalized Cauchy point + subspace minimization
# on B = θI - W M Wᵀ), but written for clarity rather than speed: dense Gram
# rebuilds and an explicit `inv` instead of Cholesky factors, and fresh
# allocations instead of in-place workspace. Run it with `Optim.SimpleLBFGSB()`.

module SimpleLBFGSBReference

import ..Optim
import NLSolversBase
using NLSolversBase: OnceDifferentiable, NonDifferentiable, value_gradient!
using LinearAlgebra: dot, norm, I, Diagonal, tril, diag
import ADTypes

abstract type LineSearcher end
include("lbfgsb/hz.jl")
include("lbfgsb/linesearches_mt.jl")

struct SimpleLBFGSB{L<:LineSearcher,T} <: Optim.AbstractConstrainedOptimizer
    m::Int
    linesearch::L
    stepsize::T
    clip_subspace::Bool
end

#=
    Optim.SimpleLBFGSB(; m=10, linesearch=HZAW(), stepsize=1.0, clip_subspace=true)

Reference implementation of L-BFGS-B, kept internal and **not exported**. It
solves the same bound-constrained problem as `Optim.LBFGSB` with the same
options and keywords, but uses dense matrix rebuilds and an explicit inverse
instead of the Cholesky-factored, incremental, in-place machinery of `LBFGSB`.
It is a fully independent implementation, kept only for cross-checking the
efficient solver, and is not intended for production use. Run it with
`Optim.SimpleLBFGSB()`. Its line search keyword accepts
`Optim.SimpleLBFGSBReference.HZAW()` / `...MTLS()`.
=#
function SimpleLBFGSB(;
    m::Integer = 10,
    linesearch::LineSearcher = HZAW(),
    stepsize::Real = 1.0,
    clip_subspace::Bool = true,
)
    SimpleLBFGSB(Int(m), linesearch, float(stepsize), clip_subspace)
end

Base.summary(io::IO, ::SimpleLBFGSB) = print(io, "L-BFGS-B (simple reference)")

# Own optimizer state for the termination code. Subtypes AbstractOptimizerState
# (not ZerothOrderState) so the generic change accessors return real values.
struct SimpleState{T,Tx} <: Optim.AbstractOptimizerState
    x::Tx
    f_x::T
    x_previous::Tx
    f_x_previous::T
end

# Dense compact representation: all Gram matrices are rebuilt each update.
mutable struct CompactLBFGS{T}
    m::Int
    S::Matrix{T}
    Y::Matrix{T}
    SᵀS::Matrix{T}
    L::Matrix{T}
    W::Matrix{T}
    M::Matrix{T}
    θ::T
    k::Int
end

function CompactLBFGS{T}(n::Integer, m::Integer) where {T}
    CompactLBFGS{T}(
        m,
        zeros(T, n, m),
        zeros(T, n, m),
        zeros(T, m, m),
        zeros(T, m, m),
        zeros(T, n, 2m),
        zeros(T, 2m, 2m),
        one(T),
        0,
    )
end

function reset_history!(B::CompactLBFGS{T}) where {T}
    B.k = 0
    B.θ = one(T)
    return B
end

project(x, lb, ub) = clamp.(x, lb, ub)

function max_steplength(x, d, lb, ub)
    T = eltype(x)
    αmax = T(Inf)
    for i in eachindex(x)
        if d[i] > 0
            αmax = min(αmax, (ub[i] - x[i]) / d[i])
        elseif d[i] < 0
            αmax = min(αmax, (lb[i] - x[i]) / d[i])
        end
    end
    return αmax
end

function pg_norm(x, g, lb, ub)
    result = zero(eltype(g))
    for i in eachindex(x, g, lb, ub)
        gi = g[i]
        if gi < 0
            isfinite(ub[i]) && (gi = max(x[i] - ub[i], gi))
        elseif gi > 0
            isfinite(lb[i]) && (gi = min(x[i] - lb[i], gi))
        end
        agi = abs(gi)
        agi > result && (result = agi)
    end
    return result
end

# Generalized Cauchy point; returns (xᶜ, c) with c = Wᵀ(xᶜ - x).
function cauchy_point(x, g, lb, ub, B::CompactLBFGS)
    T = eltype(x)
    n = length(x)
    mcols = 2B.k
    θ = B.θ

    W = @view B.W[:, 1:mcols]
    M = @view B.M[1:mcols, 1:mcols]

    t = fill(T(Inf), n)
    d = zeros(T, n)
    for i = 1:n
        if g[i] < 0
            t[i] = (x[i] - ub[i]) / g[i]
        elseif g[i] > 0
            t[i] = (x[i] - lb[i]) / g[i]
        end
        if t[i] > 0
            d[i] = -g[i]
        end
    end

    order = sortperm(t)
    p = W' * d
    c = zeros(T, mcols)
    f′ = -dot(d, d)
    f″ = -θ * f′ - dot(p, M * p)
    Δtmin = -f′ / f″
    t_old = zero(T)

    idx = 1
    while idx <= n && t[order[idx]] <= 0
        idx += 1
    end
    if idx > n
        return copy(x), zeros(T, mcols)
    end
    t_cur = t[order[idx]]
    Δt = t_cur - t_old

    while Δtmin >= Δt && idx <= n && isfinite(Δt)
        b = order[idx]

        xᶜ_b = b <= 0 ? zero(T) : (d[b] > 0 ? ub[b] : lb[b])
        z_b = xᶜ_b - x[b]

        c .+= Δt .* p
        g_b = g[b]
        wb = @view W[b, :]
        f′ += Δt * f″ + g_b^2 + θ * g_b * z_b - g_b * dot(wb, M * c)
        f″ += -θ * g_b^2 - 2 * g_b * dot(wb, M * p) - g_b^2 * dot(wb, M * wb)
        p .+= g_b .* wb
        d[b] = zero(T)

        Δtmin = f″ > 0 ? -f′ / f″ : (f′ < 0 ? T(Inf) : zero(T))
        t_old = t_cur

        idx += 1
        while idx <= n && t[order[idx]] <= t_old
            idx += 1
        end
        if idx <= n
            t_cur = t[order[idx]]
            Δt = t_cur - t_old
        else
            break
        end
    end

    Δtmin = max(Δtmin, zero(T))
    t_old += Δtmin
    xc = copy(x)
    for i = 1:n
        if g[i] == 0
            xc[i] = x[i]
        else
            xc[i] = clamp(x[i] - t_old * g[i], lb[i], ub[i])
        end
    end
    c .+= Δtmin .* p

    return xc, c
end

# Subspace minimization over the free variables at the Cauchy point.
function subspace_optimize(x, xc, c, g, lb, ub, B::CompactLBFGS; clip::Bool = true)
    T = eltype(x)
    θ = B.θ
    W = @view B.W[:, 1:2B.k]
    M = @view B.M[1:2B.k, 1:2B.k]

    free = findall(i -> lb[i] < xc[i] < ub[i], eachindex(xc))

    if isempty(free) || B.k == 0
        return copy(xc)
    end

    Wf = W[free, :]
    x̂c = xc[free] .- x[free]
    rc = g[free] .+ θ .* x̂c .- Wf * (M * c)
    v = M * (Wf' * rc)
    N = Matrix(I - (one(T) / θ) .* (M * (Wf' * Wf)))
    v = N \ v
    du = -(one(T) / θ) .* rc .- (one(T) / θ^2) .* (Wf * v)

    x̄ = copy(xc)

    if clip
        any_bound_hit = false
        for (j, i) in enumerate(free)
            x̄[i] = clamp(xc[i] + du[j], lb[i], ub[i])
            if x̄[i] == lb[i] || x̄[i] == ub[i]
                any_bound_hit = true
            end
        end

        if any_bound_hit
            dd_p = dot(x̄ .- x, g)
            if dd_p > 0
                α_star = one(T)
                for (j, i) in enumerate(free)
                    if du[j] > 0
                        α_star = min(α_star, (ub[i] - xc[i]) / du[j])
                    elseif du[j] < 0
                        α_star = min(α_star, (lb[i] - xc[i]) / du[j])
                    end
                end
                for (j, i) in enumerate(free)
                    x̄[i] = xc[i] + α_star * du[j]
                end
            end
        end
    else
        α_star = one(T)
        for (j, i) in enumerate(free)
            if du[j] > 0
                α_star = min(α_star, (ub[i] - xc[i]) / du[j])
            elseif du[j] < 0
                α_star = min(α_star, (lb[i] - xc[i]) / du[j])
            end
        end
        for (j, i) in enumerate(free)
            x̄[i] = xc[i] + α_star * du[j]
        end
    end

    return x̄
end

function update_compact!(B::CompactLBFGS, x, x_new, g, g_new)
    T = eltype(x)
    s = x_new .- x
    y = g_new .- g
    yts = dot(y, s)

    yts <= 0 && return B

    k = B.k
    m = B.m

    if k < m
        k += 1
        B.k = k
        B.S[:, k] .= s
        B.Y[:, k] .= y
    else
        for j = 1:(m-1)
            B.S[:, j] .= B.S[:, j+1]
            B.Y[:, j] .= B.Y[:, j+1]
        end
        B.S[:, m] .= s
        B.Y[:, m] .= y
    end

    B.θ = dot(y, y) / yts

    Sk = @view B.S[:, 1:k]
    Yk = @view B.Y[:, 1:k]
    SᵀY = Sk' * Yk

    B.SᵀS[1:k, 1:k] .= Sk' * Sk
    B.L[1:k, 1:k] .= tril(SᵀY, -1)

    B.W[:, 1:k] .= Yk
    B.W[:, (k+1):2k] .= B.θ .* Sk

    D = Diagonal(diag(SᵀY))
    Minv = zeros(T, 2k, 2k)
    Minv[1:k, 1:k] .= -D
    Minv[1:k, (k+1):2k] .= B.L[1:k, 1:k]'
    Minv[(k+1):2k, 1:k] .= B.L[1:k, 1:k]
    Minv[(k+1):2k, (k+1):2k] .= B.θ .* B.SᵀS[1:k, 1:k]
    B.M[1:2k, 1:2k] .= inv(Minv)

    return B
end

function trace!(tr, iteration, f_x, pgnorm, x, g, options, curr_time)
    dt = Dict{String,Any}()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(x)
        dt["g(x)"] = copy(g)
    end
    Optim.update!(
        tr,
        iteration,
        f_x,
        pgnorm,
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
    )
end

function Optim.optimize(
    f,
    l::AbstractArray,
    u::AbstractArray,
    x0::AbstractArray,
    method::SimpleLBFGSB,
    options::Optim.Options = Optim.Options();
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = Optim.DEFAULT_AD_TYPE,
)
    if f isa NonDifferentiable
        f = f.f
    end
    od = OnceDifferentiable(f, x0, zero(eltype(x0)); inplace, autodiff)
    Optim.optimize(od, l, u, x0, method, options)
end

function Optim.optimize(
    f,
    g,
    l::AbstractArray,
    u::AbstractArray,
    x0::AbstractArray,
    method::SimpleLBFGSB,
    options::Optim.Options = Optim.Options();
    inplace::Bool = true,
)
    g! = inplace ? g : (G, x) -> copyto!(G, g(x))
    od = OnceDifferentiable(f, g!, x0, zero(eltype(x0)))
    Optim.optimize(od, l, u, x0, method, options)
end

function Optim.optimize(
    d::OnceDifferentiable,
    l::AbstractArray,
    u::AbstractArray,
    x0::AbstractArray,
    method::SimpleLBFGSB,
    options::Optim.Options = Optim.Options(),
)
    T = eltype(x0)
    t0 = time()
    (; callback) = options

    n = length(x0)
    (length(l) == n && length(u) == n) ||
        throw(DimensionMismatch("lb, ub and x0 must have the same length."))
    all(i -> l[i] <= u[i], eachindex(l)) ||
        throw(ArgumentError("Each lower bound must be ≤ the corresponding upper bound."))
    all(i -> l[i] <= x0[i] <= u[i], eachindex(x0)) ||
        throw(ArgumentError("Initial x0 is outside the box [lb, ub]."))

    x = project(copy(x0), l, u)
    f_x, _g = value_gradient!(d, x)
    g = copy(_g)

    B = CompactLBFGS{T}(n, method.m)

    tr = Optim.OptimizationTrace{typeof(f_x),typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace
    options.show_trace && Optim.print_header(method)

    x_previous = copy(x)
    f_x_previous = oftype(f_x, NaN)

    pgnorm = pg_norm(x, g, l, u)

    x_converged = false
    f_converged = false
    g_converged = pgnorm <= options.g_abstol
    f_increased = false
    ls_failed = false
    stopped_by_time_limit = false
    f_limit_reached = false
    g_limit_reached = false

    iteration = 0
    _time = time()
    tracing && trace!(tr, iteration, f_x, pgnorm, x, g, options, _time - t0)
    stopped_by_callback = callback !== nothing && callback((; x, f_x, g_x = g, iteration)) == true

    converged = g_converged
    stopped = stopped_by_callback || !isfinite(f_x) || any(!isfinite, g)

    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        xc, c = cauchy_point(x, g, l, u, B)
        x̂ = subspace_optimize(x, xc, c, g, l, u, B; clip = method.clip_subspace)
        dvec = x̂ .- x

        dφ0 = dot(g, dvec)
        if dφ0 >= 0 && B.k > 0
            reset_history!(B)
            xc, c = cauchy_point(x, g, l, u, B)
            x̂ = subspace_optimize(x, xc, c, g, l, u, B; clip = method.clip_subspace)
            dvec = x̂ .- x
            dφ0 = dot(g, dvec)
        end
        if dφ0 >= 0
            ls_failed = true
            break
        end

        αmax = max_steplength(x, dvec, l, u)
        φdφ = α -> begin
            xα = x .+ α .* dvec
            fα, gα = value_gradient!(d, xα)
            (fα, dot(gα, dvec))
        end

        dnorm = norm(dvec)
        boxed = all(i -> isfinite(l[i]) && isfinite(u[i]), eachindex(x))
        α₀ = if B.k == 0 && !boxed && dnorm > 0
            min(one(T) / dnorm, αmax)
        else
            min(T(method.stepsize), αmax)
        end

        α, _, _ = find_steplength(method.linesearch, φdφ, f_x, dφ0, α₀; αmax = αmax)
        if !isfinite(α)
            α = min(T(1) / 10^4, αmax)
        end

        x_new = x .+ α .* dvec
        f_new, _gn = value_gradient!(d, x_new)
        g_new = copy(_gn)

        yts = dot(g_new .- g, x_new .- x)
        if isfinite(f_new) && yts > 0
            update_compact!(B, x, x_new, g, g_new)
        end

        copyto!(x_previous, x)
        f_x_previous = f_x
        copyto!(x, x_new)
        f_x = f_new
        copyto!(g, g_new)

        pgnorm = pg_norm(x, g, l, u)

        x_converged =
            Optim.x_abschange(x, x_previous) <= options.x_abstol ||
            Optim.x_relchange(x, x_previous) <= options.x_reltol
        f_converged =
            Optim.f_abschange(f_x, f_x_previous) <= options.f_abstol ||
            Optim.f_relchange(f_x, f_x_previous) <= options.f_reltol
        g_converged = pgnorm <= options.g_abstol
        f_increased = f_x > f_x_previous
        converged = x_converged || f_converged || g_converged

        tracing && trace!(tr, iteration, f_x, pgnorm, x, g, options, time() - t0)
        if callback !== nothing
            stopped_by_callback = callback((; x, f_x, g_x = g, iteration)) == true
        end

        _time = time()
        stopped_by_time_limit = _time - t0 > options.time_limit
        f_limit_reached =
            options.f_calls_limit > 0 && NLSolversBase.f_calls(d) >= options.f_calls_limit
        g_limit_reached =
            options.g_calls_limit > 0 && NLSolversBase.g_calls(d) >= options.g_calls_limit

        if stopped_by_callback ||
           stopped_by_time_limit ||
           f_limit_reached ||
           g_limit_reached ||
           (f_increased && !options.allow_f_increases) ||
           !isfinite(f_x) ||
           any(!isfinite, g)
            stopped = true
        end
    end

    _time = time()
    Tf = typeof(f_x)
    f_incr_pick = f_increased && !options.allow_f_increases

    stopped_by = (
        f_limit_reached = f_limit_reached,
        g_limit_reached = g_limit_reached,
        h_limit_reached = false,
        time_limit = stopped_by_time_limit,
        callback = stopped_by_callback,
        f_increased = f_incr_pick,
        ls_failed = ls_failed,
        iterations = iteration == options.iterations,
        x_converged = x_converged,
        f_converged = f_converged,
        g_converged = g_converged,
        small_trustregion_radius = false,
    )

    termination_code = Optim._termination_code(
        d,
        pgnorm,
        SimpleState(x, f_x, x_previous, f_x_previous),
        stopped_by,
        options,
    )

    return Optim.MultivariateOptimizationResults(
        method,
        x0,
        f_incr_pick ? x_previous : x,
        Tf(f_incr_pick ? f_x_previous : f_x),
        iteration,
        Tf(options.x_abstol),
        Tf(options.x_reltol),
        Optim.x_abschange(x, x_previous),
        Optim.x_relchange(x, x_previous),
        Tf(options.f_abstol),
        Tf(options.f_reltol),
        Optim.f_abschange(f_x, f_x_previous),
        Optim.f_relchange(f_x, f_x_previous),
        Tf(options.g_abstol),
        pgnorm,
        tr,
        NLSolversBase.f_calls(d),
        NLSolversBase.g_calls(d),
        NLSolversBase.jvp_calls(d),
        NLSolversBase.h_calls(d),
        NLSolversBase.hvp_calls(d),
        options.time_limit,
        _time - t0,
        stopped_by,
        termination_code,
    )
end

end # module SimpleLBFGSBReference
