# Attempt to compute a reasonable default mu: at the starting
# position, the gradient of the input function should dominate the
# gradient of the barrier.
function initialize_mu{T}(gfunc::Array{T}, gbarrier::Array{T}, ops::Options)
    mu0::T
    mu0factor::T
    @defaults ops mu0=nan(T) mu0factor=0.001
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
    @check_used ops
    return mu
end

function barrier_box{T}(g, x::Array{T}, l::Array{T}, u::Array{T})
    n = length(x)
    calc_grad = !(g === nothing)

    v = zero(T)
    for i = 1:n
        thisl = l[i]
        if isfinite(thisl)
            dx = x[i] - thisl
            if dx <= 0
                return inf(T)
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
                return inf(T)
            end
            v -= log(dx)
            if calc_grad
                g[i] += one(T)/dx
            end
        end
    end
    return v
end

function function_barrier{T}(gfunc, gbarrier, x::Array{T}, f::Function, fbarrier::Function)
    vbarrier = fbarrier(gbarrier, x)
    if isfinite(vbarrier)
        vfunc = f(gfunc, x)
    else
        vfunc = vbarrier
    end
    return vfunc, vbarrier
end

function barrier_combined{T}(g, gfunc, gbarrier, valeach::Vector{T}, x::Array{T}, fb::Function, mu::T)
    calc_grad = !(g === nothing)
    valfunc, valbarrier = fb(gfunc, gbarrier, x)
    valeach[1] = valfunc
    valeach[2] = valbarrier
    if calc_grad
        for i = 1:length(g)
            g[i] = gfunc[i] + mu*gbarrier[i]
        end
    end
    return convert(T, valfunc + mu*valbarrier) # FIXME make this unnecessary
end

function limits_box{T}(x::Array{T}, d::Array{T}, l::Array{T}, u::Array{T})
    alphamax = inf(T)
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

function fminbox{T}(func::Function, x::Array{T}, l::Array{T}, u::Array{T}, ops::Options)
    tol::T
    mufactor::T
    @defaults ops tol=eps(T)^(2/3) mufactor=0.001 display=0 optimizer=(func, x, ops)->cgdescent(func, x, ops) P=Array(T,length(x)) precondprep=precondprepbox
    ops = copy(ops)  # to avoid passing back extended options
    x = copy(x)
    fbarrier = (gbarrier, x) -> barrier_box(gbarrier, x, l, u)
    fb = (gfunc, gbarrier, x) -> function_barrier(gfunc, gbarrier, x, func, fbarrier)
    gfunc = similar(x)
    gbarrier = similar(x)
    # Because we use the gradient to estimate the initial mu, we have
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
    valfunc = func(gfunc, x)
    mu = initialize_mu(gfunc, gbarrier, ops)
    if display > 0
        println("######## fminbox ########")
        println("Initial mu = ", mu)
    end

    g = similar(x)
    valboth = Array(T, 2)    
    fval_all = Array(Vector{T}, 0)
    fcount_all = 0
    xold = similar(x)
    converged = false
    @set_options ops tol=10*tol alphamaxfunc=(x, d)->limits_box(x, d, l, u) P=P reportfunc=val->valboth[1]
    while true
        copy!(xold, x)
        # Optimize with current setting of mu
        funcc = (g, x) -> barrier_combined(g, gfunc, gbarrier, valboth, x, fb, mu)
        @set_options ops precondprep=(out, x)->precondprep(out, x, l, u, mu)
        if display > 0
            println("#### Calling optimizer with mu = ", mu, " ####")
        end
        x, fval, fcount, converged = optimizer(funcc, x, ops)
        if display & PARAMETERS_MU > 0
            println("x: ", x)
        end
        push!(fval_all, fval)
        fcount_all += fcount

        # Decrease mu
        mu *= mufactor

        # Test for convergence
        fcmp = abs(fval[1]) + abs(fval[end])
        tot = zero(T)
        for i = 1:length(x)
            tot += abs((x[i] - xold[i]) * (gfunc[i] + mu*gbarrier[i]))
        end
        if tot <= tol * (fcmp + eps(T))
            break
        end
    end
    @check_used ops
    return x, fval_all, fcount_all, converged
end
fminbox{T}(func::Function, x::Array{T}, l::Array{T}, u::Array{T}) = fminbox(func, x, l, u, Options())
export fminbox
