# Attempt to compute a reasonable default mu: at the starting
# position, the gradient of the input function should dominate the
# gradient of the barrier.
function initial_mu(gfunc::AbstractArray{T}, gbarrier::AbstractArray{T}, mu0factor::T = 0.001, mu0::T = convert(T, NaN)) where T
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

function barrier_box(g, x::AbstractArray{T}, l::AbstractArray{T}, u::AbstractArray{T}) where T
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

function function_barrier(gfunc, gbarrier, x::AbstractArray{T}, f::F, fbarrier::FB) where {T, F<:Function, FB<:Function}
    vbarrier = fbarrier(gbarrier, x)
    if isfinite(vbarrier)
        vfunc = f(gfunc, x)
    else
        vfunc = vbarrier
    end
    return vfunc, vbarrier
end

function barrier_combined(gfunc, gbarrier, g, x::AbstractArray{T}, fb::FB, mu::T) where {T, FB<:Function}
    calc_g = !(g === nothing)
    valfunc, valbarrier = fb(gbarrier, x, gfunc)
    if calc_g
        g .= gfunc .+ mu.*gbarrier
    end
    return convert(T, valfunc + mu*valbarrier) # FIXME make this unnecessary
end

function limits_box(x::AbstractArray{T}, d::AbstractArray{T}, l::AbstractArray{T}, u::AbstractArray{T}) where T
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
    @. P.diag = 1/(mu[]*(1/(x-l)^2 + 1/(u-x)^2) + 1)
end

struct Fminbox{T, Tf, P} <: AbstractConstrainedOptimizer
    method::T
    mu0::Tf
    mufactor::Tf
    precondprep::P
end
Fminbox() = Fminbox(LBFGS(); mu0 = NaN, mufactor = 0.001, precondprep = (P, x, l, u, mu) -> precondprepbox!(P, x, l, u, mu))
function Fminbox(method;
        mu0 = NaN,
        mufactor = 0.001,
        precondprep = (P, x, l, u, mu) -> precondprepbox!(P, x, l, u, mu))

Fminbox(method, promote(mu0, mufactor)..., precondprep) # default optimizer
end

Base.summary(F::Fminbox) = "Fminbox with $(summary(F.method))"

function barrier_method(F, P, mu, l, u)
    # Define the barrier-aware preconditioner once and for all (mu is mutable 1-element vector)
    pcp = (P, x) -> F.precondprep(P, x, l, u, mu)

    O = F.method
    if typeof(O) <: ConjugateGradient
        return ConjugateGradient(eta = O.eta, alphaguess = O.alphaguess!, linesearch = O.linesearch!, P = P, precondprep = pcp)
    elseif typeof(O) <: LBFGS
        return LBFGS(alphaguess = O.alphaguess!, linesearch = O.linesearch!, P = P, precondprep = pcp)
    elseif typeof(O) <: GradientDescent
        return GradientDescent(alphaguess = O.alphaguess!, linesearch = O.linesearch!, P = P, precondprep = pcp)
    elseif typeof(O) <: Union{NelderMead, SimulatedAnnealing, ParticleSwarm, BFGS, AbstractNGMRES}
        return O
    else
        error("You need to specify a valid inner optimizer, please consult the documentation.")
    end
end


function optimize(obj,
                  l::AbstractArray{T},
                  u::AbstractArray{T},
                  initial_x::AbstractArray{T},
                  F::Fminbox{O} = Fminbox()) where {T<:AbstractFloat,O<:AbstractOptimizer}
    od = OnceDifferentiable(obj, initial_x, zero(T))
    optimize(od, l, u, initial_x, F)
end

function optimize(f,
                  g!,
                  l::AbstractArray{T},
                  u::AbstractArray{T},
                  initial_x::AbstractArray{T},
                  F::Fminbox{O} = Fminbox()) where {T<:AbstractFloat,O<:AbstractOptimizer}
    od = OnceDifferentiable(f, g!, initial_x, zero(T))
    optimize(od, l, u, initial_x, F)
end

function optimize(
        df::OnceDifferentiable,
        l::AbstractArray{T},
        u::AbstractArray{T},
        initial_x::AbstractArray{T},
        F::Fminbox{O} = Fminbox(),
        options = Options()) where {T<:AbstractFloat,O<:AbstractOptimizer}


    outer_iterations = options.outer_iterations
    allow_outer_f_increases = options.allow_outer_f_increases
    show_trace, store_trace, extended_trace = options.show_trace, options.store_trace, options.extended_trace

    O == Newton && warn("Newton is not supported as the inner optimizer. Defaulting to ConjugateGradient.")
    x = copy(initial_x)
    fbarrier = (gbarrier, x) -> barrier_box(gbarrier, x, l, u)
    fb = (gbarrier, x, gfunc) -> function_barrier(gfunc, gbarrier, x, df.fdf, fbarrier)
    gfunc = similar(x)
    gbarrier = similar(x)
    P = InverseDiagonal(similar(initial_x))
    # to be careful about one special case that might occur commonly
    # in practice: the initial guess x is exactly in the center of the
    # box. In that case, gbarrier is zero. But since the
    # initialization only makes use of the magnitude, we can fix this
    # by using the sum of the absolute values of the contributions
    # from each edge.
    boundaryidx = Array{Int,1}()
    for i = 1:length(gbarrier)
        thisx = x[i]
        thisl = l[i]
        thisu = u[i]

        if thisx == thisl
            thisx = 0.99*thisl+0.01*thisu
            x[i] = thisx
            push!(boundaryidx,i)
        elseif thisx == thisu
            thisx = 0.01*thisl+0.99*thisu
            x[i] = thisx
            push!(boundaryidx,i)
        elseif thisx < thisl || thisx > thisu
            error("Initial position must be inside the box")
        end

        gbarrier[i] = (isfinite(thisl) ? one(T)/(thisx-thisl) : zero(T)) + (isfinite(thisu) ? one(T)/(thisu-thisx) : zero(T))
    end
    if length(boundaryidx) > 0
        warn("Initial position cannot be on the boundary of the box. Moving elements to the interior.\nElement indices affected: $boundaryidx")
    end

    gradient!(df, x)
    gfunc .= gradient(df)

    mu = Ref(initial_mu(gfunc, gbarrier, F.mufactor, F.mu0))

    # Create barrier-aware method instance (precondition relevance)
    _optimizer = barrier_method(F, P, mu, l, u)

    if show_trace > 0
        println("######## fminbox ########")
        println("Initial mu = ", mu[])
    end

    g = similar(x)
    fval_all = Array{Vector{T}}(0)

    # Count the total number of outer iterations
    iteration = 0

    xold = similar(x)
    converged = false
    local results
    first = true
    fval0 = zero(T)

    while !converged && iteration < outer_iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        copy!(xold, x)
        # Optimize with current setting of mu
        funcc = (g, x) -> barrier_combined(gfunc, gbarrier,  g, x, fb, mu[])
        fval0 = funcc(nothing, x)
        dfbox = OnceDifferentiable(x->funcc(nothing, x), (g, x)->(funcc(g, x); g), funcc, initial_x, zero(T))
        if show_trace > 0
            println("#### Calling optimizer with mu = ", mu[], " ####")
        end
        resultsnew = optimize(dfbox, x, _optimizer, options)
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
        mu[] *= F.mufactor

        # Test for convergence
        g .= gfunc .+ mu[].*gbarrier

        results.x_converged, results.f_converged,
        results.g_converged, converged, f_increased = assess_convergence(x, xold, minimum(results), fval0, g,
                                                                         options.outer_x_tol, options.outer_f_tol, options.outer_g_tol)
        f_increased && !allow_outer_f_increases && break
    end

    return MultivariateOptimizationResults(F, initial_x, minimizer(results), df.f(minimizer(results)),
            iteration, results.iteration_converged,
            results.x_converged, results.x_tol, vecnorm(x - xold),
            results.f_converged, results.f_tol, f_abschange(minimum(results), fval0),
            results.g_converged, results.g_tol, vecnorm(g, Inf),
            results.f_increased, results.trace, results.f_calls,
            results.g_calls, results.h_calls)
end
