# Attempt to compute a reasonable default mu: at the starting
# position, the gradient of the input function should dominate the
# gradient of the barrier.
function initial_mu{T}(gfunc::Array{T}, gbarrier::Array{T}; mu0::T = convert(T, NaN), mu0factor::T = 0.001)
    if isnan(mu0)
        gbarriernorm = sum(abs, gbarrier)
        if gbarriernorm > 0
            mu = mu0factor*sum(abs, gfunc)/gbarriernorm
        else
            # Presumably, there is no barrier function
            mu = zero(T)
        end
    else
        mu = mu0
    end
    return mu
end

function barrier_box{T}(x::Array{T}, g, l::Array{T}, u::Array{T})
    n = length(x)
    calc_g = !(g === nothing)

    v = zero(T)
    for i = 1:n
        thisl = l[i]
        if isfinite(thisl)
            dx = x[i] - thisl
            if dx <= 0
                return convert(T, Inf)
            end
            v -= log(dx)
            if calc_g
                g[i] = -one(T)/dx
            end
        else
            if calc_g
                g[i] = zero(T)
            end
        end
        thisu = u[i]
        if isfinite(thisu)
            dx = thisu - x[i]
            if dx <= 0
                return convert(T, Inf)
            end
            v -= log(dx)
            if calc_g
                g[i] += one(T)/dx
            end
        end
    end
    return v
end

function function_barrier{T, F<:Function, FB<:Function}(x::Array{T}, gfunc, gbarrier, f::F, fbarrier::FB)
    vbarrier = fbarrier(x, gbarrier)
    if isfinite(vbarrier)
        vfunc = f(x, gfunc)
    else
        vfunc = vbarrier
    end
    return vfunc, vbarrier
end

function barrier_combined{T, FB<:Function}(x::Array{T}, g, gfunc, gbarrier, fb::FB, mu::T)
    calc_g = !(g === nothing)
    valfunc, valbarrier = fb(x, gfunc, gbarrier)
    if calc_g
        @simd for i = 1:length(g)
            @inbounds g[i] = gfunc[i] + mu*gbarrier[i]
        end
    end
    return convert(T, valfunc + mu*valbarrier) # FIXME make this unnecessary
end

function limits_box{T}(x::Array{T}, d::Array{T}, l::Array{T}, u::Array{T})
    alphamax = convert(T, Inf)
    for i = 1:length(x)
        if d[i] < 0
            @inbounds alphamax = min(alphamax, ((l[i]-x[i])+eps(l[i]))/d[i])
        elseif d[i] > 0
            @inbounds alphamax = min(alphamax, ((u[i]-x[i])-eps(u[i]))/d[i])
        end
    end
    epsilon = eps(max(alphamax, one(T)))
    if !isinf(alphamax) && alphamax > epsilon
        alphamax -= epsilon
    end
    return alphamax
end

# Default preconditioner for box-constrained optimization
# This creates the inverse Hessian of the barrier penalty
function precondprepbox!(P, x, l, u, mu)
    @inbounds @simd for i = 1:length(x)
        xi = x[i]
        li = l[i]
        ui = u[i]
        P.diag[i] = 1/(mu*(1/(xi-li)^2 + 1/(ui-xi)^2) + 1) # +1 like identity far from edges
    end
end

immutable Fminbox <: Optimizer end

function optimize{T<:AbstractFloat}(
        df::DifferentiableFunction,
        initial_x::Array{T},
        l::Array{T},
        u::Array{T},
        ::Fminbox;
        x_tol::T = eps(T),
        f_tol::T = sqrt(eps(T)),
        g_tol::T = sqrt(eps(T)),
        allow_f_increases::Bool = true,
        iterations::Integer = 1_000,
        store_trace::Bool = false,
        show_trace::Bool = false,
        extended_trace::Bool = false,
        callback = nothing,
        show_every::Integer = 1,
        linesearch! = nothing,
        linesearch = LineSearches.hagerzhang!,
        eta::Real = convert(T,0.4),
        mu0::T = convert(T, NaN),
        mufactor::T = convert(T, 0.001),
        precondprep! = nothing,
        precondprep = (P, x, l, u, mu) -> precondprepbox!(P, x, l, u, mu),
        optimizer = ConjugateGradient,
        optimizer_o = Options(store_trace = store_trace,
                                          show_trace = show_trace,
                                          extended_trace = extended_trace),
        nargs...)

    # remove in v0.8.0

    linesearch = get_linesearch(linesearch!, linesearch)
    precondprep = get_precondprep(precondprep!, precondprep)

    optimizer == Newton && warning("Newton is not supported as the inner optimizer. Defaulting to ConjugateGradient.")
    x = copy(initial_x)
    fbarrier = (x, gbarrier) -> barrier_box(x, gbarrier, l, u)
    fb = (x, gfunc, gbarrier) -> function_barrier(x, gfunc, gbarrier, df.fg!, fbarrier)
    gfunc = similar(x)
    gbarrier = similar(x)
    P = InverseDiagonal(Array(T, length(initial_x)))
    # to be careful about one special case that might occur commonly
    # in practice: the initial guess x is exactly in the center of the
    # box. In that case, gbarrier is zero. But since the
    # initialization only makes use of the magnitude, we can fix this
    # by using the sum of the absolute values of the contributions
    # from each edge.
    for i = 1:length(gbarrier)
        thisx = x[i]
        thisl = l[i]
        thisu = u[i]
        if thisx < thisl || thisx > thisu
            error("Initial position must be inside the box")
        end
        gbarrier[i] = (isfinite(thisl) ? one(T)/(thisx-thisl) : zero(T)) + (isfinite(thisu) ? one(T)/(thisu-thisx) : zero(T))
    end
    df.g!(x, gfunc)
    mu = isnan(mu0) ? initial_mu(gfunc, gbarrier; mu0factor=mufactor) : mu0
    if show_trace > 0
        println("######## fminbox ########")
        println("Initial mu = ", mu)
    end

    g = similar(x)
    fval_all = Array(Vector{T}, 0)

    # Count the total number of outer iterations
    iteration = 0

    xold = similar(x)
    converged = false
    local results
    first = true
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        copy!(xold, x)
        # Optimize with current setting of mu
        funcc = (x, g) -> barrier_combined(x, g, gfunc, gbarrier, fb, mu)
        fval0 = funcc(x, nothing)
        dfbox = DifferentiableFunction(x->funcc(x,nothing), (x,g)->(funcc(x,g); g), funcc)
        if show_trace > 0
            println("#### Calling optimizer with mu = ", mu, " ####")
        end
        pcp = (P, x) -> precondprep(P, x, l, u, mu)
        if optimizer == ConjugateGradient
            _optimizer = optimizer(eta = eta, linesearch = linesearch, P = P, precondprep = pcp)
        elseif optimizer in (LBFGS, GradientDescent)
            _optimizer = optimizer(linesearch = linesearch, P = P, precondprep = pcp)
        elseif optimizer in (NelderMead, SimulatedAnnealing)
            _optimizer = optimizer()
        elseif optimizer == Newton
            _optimizer = ConjugateGradient(eta = eta, linesearch = linesearch, P = P, precondprep = pcp)
        else
            _optimizer = optimizer(linesearch = linesearch)
        end
        resultsnew = optimize(dfbox, x, _optimizer, optimizer_o)
        if first
            results = resultsnew
            first = false
        else
            append!(results, resultsnew)
        end
        copy!(x, minimizer(results))
        if show_trace > 0
            println("x: ", x)
        end

        # Decrease mu
        mu *= mufactor

        # Test for convergence
        @simd for i = 1:length(x)
            @inbounds g[i] = gfunc[i] + mu*gbarrier[i]
        end

        results.x_converged, results.f_converged, results.g_converged, converged, f_increased = assess_convergence(x, xold, minimum(results), fval0, g, x_tol, f_tol, g_tol)
        f_increased && !allow_f_increases && break
    end
    results.method = "Fminbox with $(method(results))"
    results.iterations = iteration
    results.initial_x = initial_x
    results.minimum = df.f(minimizer(results))
    results
end
