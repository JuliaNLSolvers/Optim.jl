import NLSolversBase: value, value!, gradient, gradient!, value_gradient!, value_gradient!!
####### FIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIX THE MIDDLE OF BOX CASE THAT WAS THERE
mutable struct BarrierWrapper{TO, TB, Tm, TF, TDF} <: AbstractObjective
    obj::TO
    b::TB # barrier
    mu::Tm # multipler
    Fb::TF
    Ftotal::TF
    DFb::TDF
    DFtotal::TDF
end
f_calls(obj::BarrierWrapper) = f_calls(obj.obj)
g_calls(obj::BarrierWrapper) = g_calls(obj.obj)
h_calls(obj::BarrierWrapper) = h_calls(obj.obj)
function BarrierWrapper(obj::OnceDifferentiable, mu, lower, upper)
    barrier_term = BoxBarrier(lower, upper)

    BarrierWrapper(obj, barrier_term, mu, copy(obj.F), copy(obj.F), copy(obj.DF), copy(obj.DF))
end

struct BoxBarrier{L, U}
    lower::L
    upper::U
end
# evaluates the value and gradient components comming from the log barrier 
function _barrier_term_value(x::T, l, u) where T
    dxl = x - l
    dxu = u - x
    
    if dxl <= 0 || dxu <= 0
        return T(Inf)
    end
    vl = ifelse(isfinite(dxl), -log(dxl), T(0))
    vu = ifelse(isfinite(dxu), -log(dxu), T(0))
    return vl + vu
end
function _barrier_term_gradient(x::T, l, u) where T
    dxl = x - l
    dxu = u - x
    g = zero(T)
    if isfinite(l)
        g += -one(T)/dxl
    end
    if isfinite(u)
        g += one(T)/dxu
    end
    return g 
end
function value_gradient!(bb::BoxBarrier, g, x)
    g .= _barrier_term_gradient.(x, bb.lower, bb.upper)
    value(bb, x)
end
function gradient(bb::BoxBarrier, g, x)
    g = copy(g)
    g .= _barrier_term_gradient.(x, bb.lower, bb.upper)
end
function value_gradient!!(bb::BarrierWrapper, x)
    value_gradient!!(bb.obj, x)

    bb.DFb .= _barrier_term_gradient.(x, bb.b.lower, bb.b.upper)
    bb.DFtotal .= gradient(bb.obj) .+ bb.mu .* bb.DFb

    bb.Fb = value(bb.b, x)
    bb.Ftotal = value(bb.obj) + bb.mu*bb.Fb
end
function value_gradient!(bb::BarrierWrapper, x)
    value_gradient!(bb.obj, x)

    bb.DFb .= _barrier_term_gradient.(x, bb.b.lower, bb.b.upper)
    bb.DFtotal .= gradient(bb.obj) .+ bb.mu .* bb.DFb

    bb.Fb = value(bb.b, x)
    bb.Ftotal = value(bb.obj) + bb.mu*bb.Fb
end
value(bb::BoxBarrier, x) = mapreduce(x->_barrier_term_value(x...), +, zip(x, bb.lower, bb.upper))
function value!(obj::BarrierWrapper, x)
    value!(obj.obj, x)
    obj.Fb = value(obj.b, x)
    obj.Ftotal = value(obj.obj) + obj.mu*obj.Fb
end
value(obj::BarrierWrapper) = obj.Ftotal
value(obj::BarrierWrapper, x) = value(obj.obj, x) + obj.mu*value(obj.b, x)
function gradient!(obj::BarrierWrapper, x)
    gradient!(obj.obj, x)
    obj.DFb .= gradient(obj.b, obj.DFb, x) # this should just be inplace?
    obj.DFtotal .= gradient(obj.obj) .+ obj.mu*obj.Fb
end
gradient(obj::BarrierWrapper) = obj.DFtotal
gradient(obj::BarrierWrapper, mu::Real) = gradient(obj.obj) .+ obj.mu.*obj.DFb
#gradient(obj::OnceDifferentiableBarrier, x) = gradient(obj.obj, x) + gradient(obj.b, x)

# this mutates mu but not the gradients
# Super unsafe in that it depends on x_df being correct!
function initial_mu(obj::BarrierWrapper, F)
    T = typeof(obj.Fb) # this will not work if F is real, G is complex
    gbarrier = map(x->(isfinite.(x[2]) ? one(T)/(x[1]-x[2]) : zero(T)) + (isfinite(x[3]) ? one(T)/(x[3]-x[1]) : zero(T)), zip(obj.obj.x_f, obj.b.lower, obj.b.upper))

    # obj.mu = initial_mu(gradient(obj.obj), gradient(obj.b, obj.DFb, obj.obj.x_df), T(F.mufactor), T(F.mu0))
    obj.mu = initial_mu(gradient(obj.obj), gbarrier, T(F.mufactor), T(F.mu0))
end
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
function precondprepbox!(P, x, l, u, dfbox)
    @. P.diag = 1/(dfbox.mu*(1/(x-l)^2 + 1/(u-x)^2) + 1)
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


    t0 = time()

    outer_iterations = options.outer_iterations
    allow_outer_f_increases = options.allow_outer_f_increases
    show_trace, store_trace, extended_trace = options.show_trace, options.store_trace, options.extended_trace

    x = copy(initial_x)
    P = InverseDiagonal(copy(initial_x))
    # to be careful about one special case that might occur commonly
    # in practice: the initial guess x is exactly in the center of the
    # box. In that case, gbarrier is zero. But since the
    # initialization only makes use of the magnitude, we can fix this
    # by using the sum of the absolute values of the contributions
    # from each edge.
    boundaryidx = Vector{Int}()
    for i in eachindex(l)
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
    end
    if length(boundaryidx) > 0
        @warn("Initial position cannot be on the boundary of the box. Moving elements to the interior.\nElement indices affected: $boundaryidx")
    end

    dfbox = BarrierWrapper(df, zero(T), l, u)
    # Use the barrier-aware preconditioner to define
    # barrier-aware optimization method instance (precondition relevance)
    _optimizer = barrier_method(F.method, P, (P, x) -> F.precondprep(P, x, l, u, dfbox))

    state = initial_state(_optimizer, options, dfbox, x)
    # we wait until state has been initialized to set the initial mu because
    # we need the gradient of the objective and initial_state will value_gradient!!
    # the objective, so that forces an evaluation
    dfbox.mu = initial_mu(dfbox, F)
    if show_trace > 00
        println("Fminbox")
        println("-------")
        print("Initial mu = ")
        show(IOContext(stdout, :compact=>true), "text/plain", dfbox.mu)
        println("\n")
    end

    g = copy(x)
    fval_all = Vector{Vector{T}}()

    # Count the total number of outer iterations
    iteration = 0

    # define the function (dfbox) to optimize by the inner optimizer

    xold = copy(x)
    converged = false
    local results
    first = true

    stopped = false
    _time = time()
    while !converged && !stopped && iteration < outer_iterations
        fval0 = dfbox.obj.F
        # Increment the number of steps we've had to perform
        iteration += 1

        copyto!(xold, x)
        # Optimize with current setting of mu
        if show_trace > 0
            header_string = "Fminbox iteration $iteration"
            println(header_string)
            println("-"^length(header_string))
            print("Calling inner optimizer with mu = ")
            show(IOContext(stdout, :compact=>true), "text/plain", dfbox.mu)
            println("\n")
            println("(numbers below include barrier contribution)")
        end

        # we need to update the +mu*barrier_grad part. Since we're using the
        # value_gradient! not !! as in initial_state, we won't make a superfluous
        # evaluation
        
        value_gradient!(dfbox, x)

        #if method.reset
        reset!(_optimizer, state, dfbox, x)
        #end
        resultsnew = optimize(dfbox, x, _optimizer, options, state)
        if first
            results = resultsnew
            first = false
        else
            append!(results, resultsnew)
        end
        dfbox.obj.f_calls[1] = 0
        if hasfield(typeof(dfbox.obj), :df_calls)
            dfbox.obj.df_calls[1] = 0
        end
        if hasfield(typeof(dfbox.obj), :h_calls)
            dfbox.obj.h_calls[1] = 0
        end
        copyto!(x, minimizer(results))
        boxdist = min(minimum(x-l), minimum(u-x))
        if show_trace > 0
            println()
            println("Exiting inner optimizer with x = ", x)
            print("Current distance to box: ")
            show(IOContext(stdout, :compact=>true), "text/plain", boxdist)
            println()
            println("Decreasing barrier term μ.\n")
        end
        
        # Decrease mu
        dfbox.mu *= T(F.mufactor)
        
        # Test for convergence
        g = x.-min.(max.(x.-gradient(dfbox.obj), l), u)
        results.x_converged, results.f_converged,
        results.g_converged, f_increased = assess_convergence(x, xold, minimum(results), fval0, g,
                                                                         options.outer_x_abstol, options.outer_f_reltol, options.outer_g_abstol)
        converged = results.x_converged || results.f_converged || results.g_converged
        if f_increased && !allow_outer_f_increases
            @warn("f(x) increased: stopping optimization")
            break
        end
        _time = time()
        stopped_by_time_limit = _time-t0 > options.time_limit ? true : false
        stopped = stopped_by_time_limit
    end

    return MultivariateOptimizationResults(F, initial_x, minimizer(results), df.f(minimizer(results)),
            iteration, results.iteration_converged,
            results.x_converged, results.x_abstol, results.x_reltol, norm(x - xold), norm(x - xold)/norm(x),
            results.f_converged, results.f_abstol, results.f_reltol, f_abschange(minimum(results), value(dfbox)), f_relchange(minimum(results), value(dfbox)),
            results.g_converged, results.g_abstol, norm(g, Inf),
            results.f_increased, results.trace, results.f_calls,
            results.g_calls, results.h_calls, nothing,
            options.time_limit,
            _time-t0,)
end
