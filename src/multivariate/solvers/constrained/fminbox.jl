# Attempt to compute a reasonable default mu: at the starting
# position, the gradient of the input function should dominate the
# gradient of the barrier.
function initial_mu(gfunc::AbstractArray{T}, gbarrier::AbstractArray{T}, mu0factor::T = T(1)/1000, mu0::T = convert(T, NaN)) where T
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
    calc_g = g !== nothing

    v = zero(T)
    @inbounds for i in eachindex(x)
        thisl = l[i]
        if isfinite(thisl)
            dx = x[i] - thisl
            if dx <= zero(T)
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
            if dx <= zero(T)
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

function function_barrier(gfunc, gbarrier, x::AbstractArray, f, fbarrier)
    vbarrier = fbarrier(gbarrier, x)
    return (isfinite(vbarrier) ? f(gfunc, x) : vbarrier), vbarrier
end

function barrier_combined(gfunc, gbarrier, g, x::AbstractArray,
                          fb, mu::Ref{<:Real})
    valfunc, valbarrier = fb(gbarrier, x, gfunc)
    if g !== nothing
        g .= gfunc .+ mu[].*gbarrier
    end
    return convert(eltype(x), valfunc + mu[]*valbarrier) # FIXME make this unnecessary
end

function limits_box(x::AbstractArray{T}, d::AbstractArray{T},
                    l::AbstractArray{T}, u::AbstractArray{T}) where T
    alphamax = convert(T, Inf)
    @inbounds for i in eachindex(x)
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
function precondprepbox!(P, x, l, u, mu)
    @. P.diag = 1/(mu[]*(1/(x-l)^2 + 1/(u-x)^2) + 1)
end

struct Fminbox{O<:AbstractOptimizer, T, P} <: AbstractConstrainedOptimizer
    method::O
    mu0::T
    mufactor::T
    precondprep::P
end

"""
# Fminbox
## Constructor
```julia
Fminbox(method;
        mu0=NaN,
        mufactor=0.0001,
        precondprep(P, x, l, u, mu) -> precondprepbox!(P, x, l, u, mu))
```
"""
function Fminbox(method::AbstractOptimizer = LBFGS();
                 mu0::Real = NaN, mufactor::Real = 0.001,
                 precondprep = (P, x, l, u, mu) -> precondprepbox!(P, x, l, u, mu))
    if method isa Newton || method isa NewtonTrustRegion
        throw(ArgumentError("Newton is not supported as the Fminbox optimizer."))
    end
    Fminbox(method, promote(mu0, mufactor)..., precondprep) # default optimizer
end

Base.summary(F::Fminbox) = "Fminbox with $(summary(F.method))"

# barrier_method() constructs an optimizer to solve the barrier problem using m = Fminbox.method as the reference.
# Essentially it only updates the P and precondprep fields of `m`.

# fallback
barrier_method(m::AbstractOptimizer, P, precondprep) =
    error("You need to specify a valid inner optimizer for Fminbox, $m is not supported. Please consult the documentation.")

barrier_method(m::ConjugateGradient, P, precondprep) =
    ConjugateGradient(eta = m.eta, alphaguess = m.alphaguess!,
                      linesearch = m.linesearch!, P = P,
                      precondprep = precondprep)

barrier_method(m::LBFGS, P, precondprep) =
    LBFGS(alphaguess = m.alphaguess!, linesearch = m.linesearch!, P = P,
          precondprep = precondprep)

barrier_method(m::GradientDescent, P, precondprep) =
    GradientDescent(alphaguess = m.alphaguess!, linesearch = m.linesearch!, P = P,
                    precondprep = precondprep)

barrier_method(m::Union{NelderMead, SimulatedAnnealing, ParticleSwarm, BFGS, AbstractNGMRES},
               P, precondprep) = m # use `m` as is

function optimize(f,
                  g,
                  l::AbstractArray{T},
                  u::AbstractArray{T},
                  initial_x::AbstractArray{T},
                  F::Fminbox = Fminbox(),
                  options = Options(); inplace = true, autodiff = :finite) where T<:AbstractFloat

    g! = inplace ? g : (G, x) -> copyto!(G, g(x))
    od = OnceDifferentiable(f, g!, initial_x, zero(T))

    optimize(od, l, u, initial_x, F, options)
end

optimize(f, l::Number, u::Number, initial_x::AbstractArray{T}; kwargs...) where T = optimize(f, Fill(T(l), size(initial_x)...), Fill(T(u), size(initial_x)...), initial_x; kwargs...)
optimize(f, l::AbstractArray, u::Number, initial_x::AbstractArray{T}; kwargs...) where T = optimize(f, l, Fill(T(u), size(initial_x)...), initial_x; kwargs...)
optimize(f, l::Number, u::AbstractArray, initial_x::AbstractArray{T}; kwargs...) where T = optimize(f, Fill(T(l), size(initial_x)...), u, initial_x; kwargs...)

optimize(f, l::Number, u::Number, initial_x::AbstractArray{T}, mo::AbstractConstrainedOptimizer, opt::Options=Options(); kwargs...) where T = optimize(f, Fill(T(l), size(initial_x)...), Fill(T(u), size(initial_x)...), initial_x, mo, opt; kwargs...)
optimize(f, l::AbstractArray, u::Number, initial_x::AbstractArray{T}, mo::AbstractConstrainedOptimizer, opt::Options=Options(); kwargs...) where T = optimize(f, l, Fill(T(u), size(initial_x)...), initial_x, mo, opt; kwargs...)
optimize(f, l::Number, u::AbstractArray, initial_x::AbstractArray{T}, mo::AbstractConstrainedOptimizer, opt::Options=Options(); kwargs...) where T = optimize(f, Fill(T(l), size(initial_x)...), u, initial_x, mo, opt; kwargs...)

optimize(f, g, l::Number, u::Number, initial_x::AbstractArray{T}, opt::Options; kwargs...) where T = optimize(f, g, Fill(T(l), size(initial_x)...), Fill(T(u), size(initial_x)...), initial_x, opt; kwargs...)
optimize(f, g, l::AbstractArray, u::Number, initial_x::AbstractArray{T}, opt::Options; kwargs...) where T = optimize(f, g, l, Fill(T(u), size(initial_x)...), initial_x, opt; kwargs...)
optimize(f, g, l::Number, u::AbstractArray, initial_x::AbstractArray{T}, opt::Options; kwargs...) where T = optimize(f, g, Fill(T(l), size(initial_x)...), u, initial_x, opt; kwargs...)

function optimize(f,
                  l::AbstractArray{T},
                  u::AbstractArray{T},
                  initial_x::AbstractArray{T},
                  F::Fminbox = Fminbox(),
                  options = Options(); inplace = true, autodiff = :finite) where T<:AbstractFloat

    od = OnceDifferentiable(f, initial_x, zero(T); autodiff = autodiff)
    optimize(od, l, u, initial_x, F, options)
end

function optimize(
        df::OnceDifferentiable,
        l::AbstractArray{T},
        u::AbstractArray{T},
        initial_x::AbstractArray{T},
        F::Fminbox = Fminbox(),
        options = Options()) where T<:AbstractFloat

    outer_iterations = options.outer_iterations
    allow_outer_f_increases = options.allow_outer_f_increases
    show_trace, store_trace, extended_trace = options.show_trace, options.store_trace, options.extended_trace

    x = copy(initial_x)
    fbarrier = (gbarrier, x) -> barrier_box(gbarrier, x, l, u)
    fb = (gbarrier, x, gfunc) -> function_barrier(gfunc, gbarrier, x, df.fdf, fbarrier)
    gfunc = copy(x)
    gbarrier = copy(x)
    P = InverseDiagonal(copy(initial_x))
    # to be careful about one special case that might occur commonly
    # in practice: the initial guess x is exactly in the center of the
    # box. In that case, gbarrier is zero. But since the
    # initialization only makes use of the magnitude, we can fix this
    # by using the sum of the absolute values of the contributions
    # from each edge.
    boundaryidx = Vector{Int}()
    for i in eachindex(gbarrier)
        thisx = x[i]
        thisl = l[i]
        thisu = u[i]

        if thisx == thisl
            thisx = T(99)/100*thisl + T(1)/100*thisu
            x[i] = thisx
            push!(boundaryidx,i)
        elseif thisx == thisu
            thisx = T(1)/100*thisl + T(99)/100*thisu
            x[i] = thisx
            push!(boundaryidx,i)
        elseif thisx < thisl || thisx > thisu
            throw(ArgumentError("Initial x[$(Tuple(CartesianIndices(x)[i]))]=$thisx is outside of [$thisl, $thisu]"))
        end

        gbarrier[i] = (isfinite(thisl) ? one(T)/(thisx-thisl) : zero(T)) + (isfinite(thisu) ? one(T)/(thisu-thisx) : zero(T))
    end
    if length(boundaryidx) > 0
        @warn("Initial position cannot be on the boundary of the box. Moving elements to the interior.\nElement indices affected: $boundaryidx")
    end

    gradient!(df, x)
    gfunc .= gradient(df)

    mu = Ref(initial_mu(gfunc, gbarrier, T(F.mufactor), T(F.mu0)))

    # Use the barrier-aware preconditioner to define
    # barrier-aware optimization method instance (precondition relevance)
    _optimizer = barrier_method(F.method, P, (P, x) -> F.precondprep(P, x, l, u, mu))

    if show_trace > 0
        println("Fminbox")
        println("-------")
        print("Initial mu = ")
        show(IOContext(stdout, :compact=>true), "text/plain", mu[])
        println("\n")
    end

    g = copy(x)
    fval_all = Vector{Vector{T}}()

    # Count the total number of outer iterations
    iteration = 0

    # define the function (dfbox) to optimize by the inner optimizer
    funcc = (g, x) -> barrier_combined(gfunc, gbarrier, g, x, fb, mu)
    dfbox = OnceDifferentiable(x -> funcc(nothing, x),
                               (g, x) -> (funcc(g, x); g),
                               funcc, initial_x, zero(T))

    xold = copy(x)
    converged = false
    local results
    first = true
    fval0 = zero(T)

    while !converged && iteration < outer_iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        copyto!(xold, x)
        # Optimize with current setting of mu
        fval0 = funcc(nothing, x)
        if show_trace > 0
            header_string = "Fminbox iteration $iteration"
            println(header_string)
            println("-"^length(header_string))
            print("Calling inner optimizer with mu = ")
            show(IOContext(stdout, :compact=>true), "text/plain", mu[])
            println("\n")
            println("(numbers below include barrier contribution)")
        end
        resultsnew = optimize(dfbox, x, _optimizer, options)
        if first
            results = resultsnew
            first = false
        else
            append!(results, resultsnew)
        end
        copyto!(x, minimizer(results))
        if show_trace > 0
            println()
            println("Exiting inner optimizer with x = ", x)
            print("Current distance to box: ")
            show(IOContext(stdout, :compact=>true), "text/plain", min(minimum(x-l), minimum(u-x)))
            println()
            println("Decreasing barrier term μ.\n")
        end

        # Decrease mu
        mu[] *= T(F.mufactor)

        # Test for convergence
        g .= gfunc .+ mu[].*gbarrier

        results.x_converged, results.f_converged,
        results.g_converged, converged, f_increased = assess_convergence(x, xold, minimum(results), fval0, g,
                                                                         options.outer_x_abstol, options.outer_f_reltol, options.outer_g_abstol)
        if f_increased && !allow_outer_f_increases
            @warn("f(x) increased: stopping optimization")
            break
        end
    end
    return MultivariateOptimizationResults(F, initial_x, minimizer(results), df.f(minimizer(results)),
            iteration, results.iteration_converged,
            results.x_converged, results.x_abstol, results.x_reltol, norm(x - xold), norm(x - xold)/norm(x),
            results.f_converged, results.f_abstol, results.f_reltol, f_abschange(minimum(results), fval0), f_relchange(minimum(results), fval0),
            results.g_converged, results.g_abstol, norm(g, Inf),
            results.f_increased, results.trace, results.f_calls,
            results.g_calls, results.h_calls, nothing)
end
