# Attempt to compute a reasonable default mu: at the starting
# position, the gradient of the input function should dominate the
# gradient of the barrier.
function initialize_mu{T}(gfunc::Array{T}, gbarrier::Array{T}; mu0::T = convert(T, NaN), mu0factor::T = 0.001)
    if isnan(mu0)
        gbarriernorm = sum(abs(gbarrier))
        if gbarriernorm > 0
            mu = mu0factor*sum(abs(gfunc))/gbarriernorm
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
    calc_grad = !(g === nothing)

    v = zero(T)
    for i = 1:n
        thisl = l[i]
        if isfinite(thisl)
            dx = x[i] - thisl
            if dx <= 0
                return convert(T, Inf)
            end
            v -= log(dx)
            if calc_grad
                g[i] = -one(T)/dx
            end
        else
            if calc_grad
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
            if calc_grad
                g[i] += one(T)/dx
            end
        end
    end
    return v
end

function function_barrier{T}(x::Array{T}, gfunc, gbarrier, f::Function, fbarrier::Function)
    vbarrier = fbarrier(x, gbarrier)
    if isfinite(vbarrier)
        vfunc = f(x, gfunc)
    else
        vfunc = vbarrier
    end
    return vfunc, vbarrier
end

function barrier_combined{T}(x::Array{T}, g, gfunc, gbarrier, val_each::Vector{T}, fb::Function, mu::T)
    calc_grad = !(g === nothing)
    valfunc, valbarrier = fb(x, gfunc, gbarrier)
    val_each[1] = valfunc
    val_each[2] = valbarrier
    if calc_grad
        for i = 1:length(g)
            g[i] = gfunc[i] + mu*gbarrier[i]
        end
    end
    return convert(T, valfunc + mu*valbarrier) # FIXME make this unnecessary
end

function limits_box{T}(x::Array{T}, d::Array{T}, l::Array{T}, u::Array{T})
    alphamax = convert(T, Inf)
    for i = 1:length(x)
        if d[i] < 0
            alphamax = min(alphamax, ((l[i]-x[i])+eps(l[i]))/d[i])
        elseif d[i] > 0
            alphamax = min(alphamax, ((u[i]-x[i])-eps(u[i]))/d[i])
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
function precondprepbox(P, x, l, u, mu)
    for i = 1:length(x)
        xi = x[i]
        li = l[i]
        ui = u[i]
        P[i] = 1/(mu*(1/(xi-li)^2 + 1/(ui-xi)^2) + 1) # +1 like identity far from edges
    end
end

const PARAMETERS_MU = one64<<display_nextbit
display_nextbit += 1

function fminbox{T<:FloatingPoint}(df::DifferentiableFunction,
                    initial_x::Array{T},
                    l::Array{T},
                    u::Array{T};
                    xtol::T = eps(T),
                    ftol::T = sqrt(eps(T)),
                    grtol::T = sqrt(eps(T)),
                    iterations::Integer = 1_000,
                    store_trace::Bool = false,
                    show_trace::Bool = false,
                    extended_trace::Bool = false,
                    callback = nothing,
                    show_every = 1,
                    linesearch!::Function = hz_linesearch!,
                    eta::Real = convert(T,0.4),
                    mu0::T = convert(T, NaN),
                    mufactor::T = convert(T, 0.001),
                    precondprep = (P, x, l, u, mu) -> precondprepbox(P, x, l, u, mu),
                    optimizer = cg)

    x = copy(initial_x)
    fbarrier = (x, gbarrier) -> barrier_box(x, gbarrier, l, u)
    fb = (x, gfunc, gbarrier) -> function_barrier(x, gfunc, gbarrier, df.fg!, fbarrier)
    gfunc = similar(x)
    gbarrier = similar(x)
    P = Array(T, length(initial_x))
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
    valfunc = df.fg!(x, gfunc)  # is this used??
    mu = isnan(mu0) ? initialize_mu(gfunc, gbarrier; mu0factor=mufactor) : mu0
    if show_trace > 0
        println("######## fminbox ########")
        println("Initial mu = ", mu)
    end

    g = similar(x)
    valboth = Array(T, 2)
    fval_all = Array(Vector{T}, 0)
    fcount_all = 0
    xold = similar(x)
    converged = false
    local results
    first = true
    while true
        copy!(xold, x)
        # Optimize with current setting of mu
        funcc = (x, g) -> barrier_combined(x, g, gfunc, gbarrier, valboth, fb, mu)
        fval0 = funcc(x, nothing)
        dfbox = DifferentiableFunction(x->funcc(x,nothing), (x,g)->(funcc(x,g); g), funcc)
        if show_trace > 0
            println("#### Calling optimizer with mu = ", mu, " ####")
        end
        pcp = (P, x) -> precondprep(P, x, l, u, mu)
        resultsnew = optimizer(dfbox, x; xtol=xtol, ftol=ftol, grtol=grtol, iterations=iterations,
                                         store_trace=store_trace, show_trace=show_trace, extended_trace=extended_trace,
                                         linesearch! = linesearch!, eta=eta, P=P, precondprep=pcp)
        if first == true
            results = resultsnew
        else
            append!(results, resultsnew)
        end
        copy!(x, results.minimum)
        if show_trace > 0
            println("x: ", x)
        end

        # Decrease mu
        mu *= mufactor

        # Test for convergence
        for i = 1:length(x)
            g[i] = gfunc[i] + mu*gbarrier[i]
        end
        x_converged, f_converged, gr_converged, converged = assess_convergence(x, xold, results.f_minimum, fval0, g, xtol, ftol, grtol)
        if converged
            break
        end
    end
    results.initial_x = initial_x
    results
end
export fminbox
