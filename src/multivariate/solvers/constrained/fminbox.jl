# Attempt to compute a reasonable default mu: at the starting
# position, the gradient of the input function should dominate the
# gradient of the barrier.
function initial_mu(gfunc::AbstractArray{T}, gbarrier::AbstractArray{T}; mu0::T = convert(T, NaN), mu0factor::T = 0.001) where T
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
    T = eltype(x)
    @. P.diag = 1/(mu*(1/max(x-l, sqrt(realmin(T)))^2 + 1/max(u-x, sqrt(realmin(T)))^2) + 1)
end

struct Fminbox{T<:AbstractOptimizer} <: AbstractOptimizer end
Fminbox() = Fminbox{ConjugateGradient}() # default optimizer

Base.summary(::Fminbox{O}) where {O} = "Fminbox with $(summary(O(T)))"

function optimize(obj,
                  initial_x::AbstractArray{T},
                  l::AbstractArray{T},
                  u::AbstractArray{T},
                  F::Fminbox{O} = Fminbox(); kwargs...) where {T<:AbstractFloat,O<:AbstractOptimizer}
     optimize(OnceDifferentiable(obj, initial_x, zero(T)), initial_x, l, u, F; kwargs...)
end

function optimize(f,
                  g!,
                  initial_x::AbstractArray{T},
                  l::AbstractArray{T},
                  u::AbstractArray{T},
                  F::Fminbox{O} = Fminbox(); kwargs...) where {T<:AbstractFloat,O<:AbstractOptimizer}
     optimize(OnceDifferentiable(f, g!, initial_x, zero(T)), initial_x, l, u, F; kwargs...)
end

function optimize(
        df::OnceDifferentiable,
        initial_x::AbstractArray{T},
        l::AbstractArray{T},
        u::AbstractArray{T},
        ::Fminbox{O} = Fminbox();
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
        alphaguess = nothing,
        linesearch = nothing,
        eta::Real = convert(T,0.4),
        mu0::T = convert(T, NaN),
        mufactor::T = convert(T, 0.001),
        precondprep = (P, x, l, u, mu) -> precondprepbox!(P, x, l, u, mu),
        optimizer_o = Options(store_trace = store_trace,
                                          show_trace = show_trace,
                                          extended_trace = extended_trace),
        nargs...) where {T<:AbstractFloat,O<:AbstractOptimizer}

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
            thisx = T(99)/100*thisl+T(1)/100*thisu
            x[i] = thisx
            push!(boundaryidx,i)
        elseif thisx == thisu
            thisx = T(1)/100*thisl+T(99)/100*thisu
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
    df.df(gfunc, x)
    mu = isnan(mu0) ? initial_mu(gfunc, gbarrier; mu0factor=mufactor) : mu0
    if show_trace > 0
        println("######## fminbox ########")
        println("Initial mu = ", mu)
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
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        copy!(xold, x)
        # Optimize with current setting of mu
        funcc = (g, x) -> barrier_combined(gfunc, gbarrier,  g, x, fb, mu)
        fval0 = funcc(nothing, x)
        dfbox = OnceDifferentiable(x->funcc(nothing, x), (g, x)->(funcc(g, x); g), funcc, initial_x, zero(T))
        if show_trace > 0
            println("#### Calling optimizer with mu = ", mu, " ####")
        end
        pcp = (P, x) -> precondprep(P, x, l, u, mu)
        # TODO: Changing the default linesearch and alphaguesses
        #       in the optimization algorithms will imply a lot of extra work here
        if O == ConjugateGradient || O == Newton
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialHagerZhang{T}()
            end
            _optimizer = ConjugateGradient(eta = eta, alphaguess = alphaguess,
                                           linesearch = linesearch, P = P, precondprep = pcp)
        elseif O == LBFGS
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialStatic{T}()
            end
            _optimizer = O(T, alphaguess = alphaguess, linesearch = linesearch, P = P, precondprep = pcp)
        elseif O == BFGS
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialStatic{T}()
            end
            _optimizer = O(T, alphaguess = alphaguess, linesearch = linesearch)
        elseif O == GradientDescent
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialPrevious(alpha=one(T), alphamin=zero(T), alphamax=T(Inf))
            end
            _optimizer = O(T, alphaguess = alphaguess, linesearch = linesearch, P = P, precondprep = pcp)
        elseif O in (NelderMead, SimulatedAnnealing)
            _optimizer = O(T)
        else
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialPrevious{T}()
            end
            _optimizer = O(T, alphaguess = alphaguess, linesearch = linesearch)
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
        g .= gfunc .+ mu.*gbarrier

        results.x_converged, results.f_converged, results.g_converged, converged, f_increased = assess_convergence(x, xold, minimum(results), fval0, g, x_tol, f_tol, g_tol)
        f_increased && !allow_f_increases && break
    end

    Tx = typeof(x)
    _x_abschange = maxdiff(x,xold)
    _minimizer = minimizer(results)
    _minimum = minimum(results)
    _results::MultivariateOptimizationResults{Fminbox{O}, T, Tx, typeof(_x_abschange), typeof(_minimum), typeof(results.trace)} = MultivariateOptimizationResults{Fminbox{O}, T, Tx, typeof(_x_abschange), typeof(_minimum), typeof(results.trace)}(
        Fminbox{O}(), false, initial_x, _minimizer, _minimum,
            iteration, results.iteration_converged,
        results.x_converged, results.x_tol, _x_abschange,
        results.f_converged, results.f_tol, f_abschange(_minimum, fval0),
        results.g_converged, results.g_tol, maximum(abs, g),
            results.f_increased, results.trace, results.f_calls,
            results.g_calls, results.h_calls)
    return _results
end
