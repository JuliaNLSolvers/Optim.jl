using NLSolversBase:
    value, value!, gradient!, value_gradient!
####### FIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIX THE MIDDLE OF BOX CASE THAT WAS THERE
mutable struct BarrierWrapper{TO,TB,Tm,TF,TDF} <: AbstractObjective
    obj::TO
    b::TB # barrier
    mu::Tm # multipler
    Fb::TF
    Ftotal::TF
    DFb::TDF
    DFtotal::TDF
end

NLSolversBase.f_calls(obj::BarrierWrapper) = NLSolversBase.f_calls(obj.obj)
NLSolversBase.g_calls(obj::BarrierWrapper) = NLSolversBase.g_calls(obj.obj)
NLSolversBase.h_calls(obj::BarrierWrapper) = NLSolversBase.h_calls(obj.obj)
NLSolversBase.hv_calls(obj::BarrierWrapper) = NLSolversBase.hv_calls(obj.obj)

function BarrierWrapper(obj::NonDifferentiable, mu, lower, upper)
    barrier_term = BoxBarrier(lower, upper)

    BarrierWrapper(obj, barrier_term, mu, oftype(obj.F, NaN), oftype(obj.F, NaN), nothing, nothing)
end
function BarrierWrapper(obj::OnceDifferentiable, mu, lower, upper)
    barrier_term = BoxBarrier(lower, upper)

    BarrierWrapper(
        obj,
        barrier_term,
        mu,
        oftype(obj.F, NaN),
        oftype(obj.F, NaN),
        fill!(copy(obj.DF), NaN),
        fill!(copy(obj.DF), NaN),
    )
end

struct BoxBarrier{L,U}
    lower::L
    upper::U
end
function in_box(bb::BoxBarrier, x)
    all(x -> x[1] >= x[2] && x[1] <= x[3], zip(x, bb.lower, bb.upper))
end
in_box(bw::BarrierWrapper, x) = in_box(bw.b, x)
# evaluates the value and gradient components comming from the log barrier
function _barrier_term_value(x::T, l, u) where {T}
    dxl = x - l
    dxu = u - x

    if dxl <= 0 || dxu <= 0
        return T(Inf)
    end
    vl = ifelse(isfinite(dxl), -log(dxl), T(0))
    vu = ifelse(isfinite(dxu), -log(dxu), T(0))
    return vl + vu
end
_barrier_value(bb::BoxBarrier, x) =
    mapreduce(x -> _barrier_term_value(x...), +, zip(x, bb.lower, bb.upper))

function _barrier_term_gradient(x::T, l, u) where {T}
    dxl = x - l
    dxu = u - x
    g = zero(T)
    if isfinite(l)
        g += -one(T) / dxl
    end
    if isfinite(u)
        g += one(T) / dxu
    end
    return g
end

# Wrappers
function NLSolversBase.value_gradient!(bb::BarrierWrapper, x)
    bb.DFb .= _barrier_term_gradient.(x, bb.b.lower, bb.b.upper)
    bb.Fb = _barrier_value(bb.b, x)
    if in_box(bb, x)
        F, DF = value_gradient!(bb.obj, x)
        bb.DFtotal .= muladd.(bb.mu, bb.DFb, DF)
        bb.Ftotal = muladd(bb.mu, bb.Fb, F)
    else
        bb.DFtotal .= bb.mu .* bb.DFb
        bb.Ftotal = bb.mu * bb.Fb
    end
    return bb.Ftotal, bb.DFtotal
end
function NLSolversBase.value!(obj::BarrierWrapper, x)
    obj.Fb = _barrier_value(obj.b, x)
    if in_box(obj, x)
        F = value!(obj.obj, x)
        obj.Ftotal = muladd(obj.mu, obj.Fb, F)
    else
        obj.Ftotal = obj.mu * obj.Fb
    end
    return obj.Ftotal
end
function NLSolversBase.value(obj::BarrierWrapper, x)
    Fb = _barrier_value(obj.b, x)
    if in_box(obj, x)
        return muladd(obj.mu, Fb, value(obj.obj, x))
    else
        return obj.mu * Fb
    end
end
function NLSolversBase.gradient!(obj::BarrierWrapper, x)
    obj.DFb .= _barrier_term_gradient.(x, obj.b.lower, obj.b.upper)
    if in_box(obj, x)
        DF = gradient!(obj.obj, x)
        obj.DFtotal .= muladd.(obj.mu, obj.DFb, DF)
    else
        obj.DFtotal .= obj.mu .* obj.DFb
    end
    return obj.DFtotal
end

function limits_box(
    x::AbstractArray{T},
    d::AbstractArray{T},
    l::AbstractArray{T},
    u::AbstractArray{T},
) where {T}
    alphamax = convert(T, Inf)
    @inbounds for i in eachindex(x)
        if d[i] < 0
            alphamax = min(alphamax, ((l[i] - x[i]) + eps(l[i])) / d[i])
        elseif d[i] > 0
            alphamax = min(alphamax, ((u[i] - x[i]) - eps(u[i])) / d[i])
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
    @. P.diag = 1 / (dfbox.mu * (1 / (x - l)^2 + 1 / (u - x)^2) + 1)
end

struct Fminbox{O<:AbstractOptimizer,T,P} <: AbstractConstrainedOptimizer
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
## Description
Fminbox implements a primal barrier method for optimization with simple
bounds (or box constraints). A description of an approach very close to
the one implemented here can be found in section 19.6 of Nocedal and Wright
 (sec. 19.6, 2006).
## References
 - Wright, S. J. and J. Nocedal (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
function Fminbox(
    method::AbstractOptimizer = LBFGS();
    mu0::Real = NaN,
    mufactor::Real = 0.001,
    precondprep = (P, x, l, u, mu) -> precondprepbox!(P, x, l, u, mu),
)
    if method isa Newton || method isa NewtonTrustRegion
        throw(ArgumentError("Newton is not supported as the Fminbox optimizer."))
    end
    Fminbox(method, promote(mu0, mufactor)..., precondprep) # default optimizer
end

function Base.summary(io::IO, F::Fminbox)
    print(io, "Fminbox with ")
    summary(io, F.method)
    return
end

# barrier_method() constructs an optimizer to solve the barrier problem using m = Fminbox.method as the reference.
# Essentially it only updates the P and precondprep fields of `m`.

# fallback
barrier_method(m::AbstractOptimizer, P, precondprep) = error(
    "You need to specify a valid inner optimizer for Fminbox, $m is not supported. Please consult the documentation.",
)

barrier_method(m::ConjugateGradient, P, precondprep) = ConjugateGradient(
    eta = m.eta,
    alphaguess = m.alphaguess!,
    linesearch = m.linesearch!,
    P = P,
    precondprep = precondprep,
)

barrier_method(m::LBFGS, P, precondprep) = LBFGS(
    alphaguess = m.alphaguess!,
    linesearch = m.linesearch!,
    P = P,
    precondprep = precondprep,
)

barrier_method(m::GradientDescent, P, precondprep) = GradientDescent(
    alphaguess = m.alphaguess!,
    linesearch = m.linesearch!,
    P = P,
    precondprep = precondprep,
)

barrier_method(
    m::Union{NelderMead,SimulatedAnnealing,ParticleSwarm,BFGS,AbstractNGMRES},
    P,
    precondprep,
) = m # use `m` as is

struct BoxState{T,Tx} <: ZerothOrderState
    x::Tx
    f_x::T
    x_previous::Tx
    f_x_previous::T
end

# Attempt to compute a reasonable default mu: at the starting
# position, the gradient of the input function should dominate the
# gradient of the barrier. 
function initial_mu(box::BoxBarrier, x::AbstractArray, g_x::AbstractArray, F::Fminbox)
    # Compute 1-norm of gradient of input function and the gradient of the barrier
    _gnorm = sum(abs, g_x)
    _gbarrier_norm = sum(Broadcast.instantiate(Broadcast.broadcasted((xi, li, ui) -> abs(_barrier_term_gradient(xi, li, ui)), x, box.lower, box.upper)))

    gnorm, gbarrier_norm, mufactor, mu0 = promote(_gnorm, _gbarrier_norm, F.mufactor, F.mu0)
    mu = if isnan(mu0)
        if gbarrier_norm > 0
            mufactor * gnorm / gbarrier_norm
        else
            # Presumably, there is no barrier function
            zero(gnorm)
        end
    else
        mu0
    end

    return mu
end

function optimize(
    f,
    l::AbstractArray,
    u::AbstractArray,
    initial_x::AbstractArray,
    F::Fminbox = Fminbox(),
    options::Options = Options();
    inplace::Bool=true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    if f isa NonDifferentiable
        f = f.f
    end
    od = OnceDifferentiable(f, initial_x, zero(eltype(initial_x)); inplace, autodiff)
    optimize(od, l, u, initial_x, F, options)
end

function optimize(
    f,
    g,
    l::AbstractArray,
    u::AbstractArray,
    initial_x::AbstractArray,
    F::Fminbox = Fminbox(),
    options::Options = Options();
    inplace = true,
)

    g! = inplace ? g : (G, x) -> copyto!(G, g(x))
    od = OnceDifferentiable(f, g!, initial_x, zero(eltype(initial_x)))

    optimize(od, l, u, initial_x, F, options)
end

function optimize(f, l::Number, u::Number, initial_x::AbstractArray; autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE)
    T = eltype(initial_x)
    optimize(
        OnceDifferentiable(f, initial_x, zero(T); autodiff),
        Fill(T(l), size(initial_x)...),
        Fill(T(u), size(initial_x)...),
        initial_x,
        Fminbox(),
        Options(),
    )
end

function optimize(
    f,
    l::Number,
    u::Number,
    initial_x::AbstractArray,
    mo::AbstractConstrainedOptimizer,
    opt::Options = Options();
    inplace::Bool=true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    T = eltype(initial_x)
    optimize(
        f,
        Fill(T(l), size(initial_x)...),
        Fill(T(u), size(initial_x)...),
        initial_x,
        mo,
        opt;
        inplace,
        autodiff,
    )
end
function optimize(
    f,
    l::AbstractArray,
    u::Number,
    initial_x::AbstractArray,
    mo::AbstractConstrainedOptimizer = Fminbox(),
    opt::Options = Options();
    inplace::Bool=true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
  T = eltype(initial_x)
optimize(f, T.(l), Fill(T(u), size(initial_x)...), initial_x, mo, opt; inplace, autodiff)
end
function optimize(
    f,
    l::Number,
    u::AbstractArray,
    initial_x::AbstractArray,
    mo::AbstractConstrainedOptimizer=Fminbox(),
    opt::Options = Options();
    inplace::Bool=true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    T = eltype(initial_x)
    optimize(f, Fill(T(l), size(initial_x)...), T.(u), initial_x, mo, opt; inplace, autodiff)
end
function optimize(
    f,
    g,
    l::Number,
    u::Number,
    initial_x::AbstractArray,
    opt::Options;
    inplace::Bool=true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
) 

T = eltype(initial_x)
optimize(
    f,
    g,
    Fill(T(l), size(initial_x)...),
    Fill(T(u), size(initial_x)...),
    initial_x,
    Fminbox(),
    opt;
    inplace,
    autodiff,
)
end
function optimize(
    f,
    g,
    l::AbstractArray,
    u::Number,
    initial_x::AbstractArray,
    opt::Options;
    inplace::Bool=true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
T = eltype(initial_x)
optimize(f, g, T.(l), Fill(T(u), size(initial_x)...), initial_x, opt; inplace, autodiff)
end

function optimize(
    f,
    g,
    l::Number,
    u::AbstractArray,
    initial_x::AbstractArray,
    opt::Options;
    inplace::Bool=true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    T= eltype(initial_x)
    optimize(f, g, Fill(T(l), size(initial_x)...), T.(u), initial_x, opt; inplace, autodiff)
end

function optimize(
    df::OnceDifferentiable,
    l::AbstractArray,
    u::AbstractArray,
    initial_x::AbstractArray,
    F::Fminbox = Fminbox(),
    options::Options = Options(),
)

    T = eltype(initial_x)
    t0 = time()

    outer_iterations = options.outer_iterations
    allow_outer_f_increases = options.allow_outer_f_increases
    show_trace, store_trace, extended_trace =
        options.show_trace, options.store_trace, options.extended_trace

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
            thisx = T(99) / 100 * thisl + T(1) / 100 * thisu
            x[i] = thisx
            push!(boundaryidx, i)
        elseif thisx == thisu
            thisx = T(1) / 100 * thisl + T(99) / 100 * thisu
            x[i] = thisx
            push!(boundaryidx, i)
        elseif thisx < thisl || thisx > thisu
            throw(
                ArgumentError(
                    "Initial x[$(Tuple(CartesianIndices(x)[i]))]=$thisx is outside of [$thisl, $thisu]",
                ),
            )
        end
    end
    if length(boundaryidx) > 0
        @warn(
            "Initial position cannot be on the boundary of the box. Moving elements to the interior.\nElement indices affected: $boundaryidx"
        )
    end

    dfbox = BarrierWrapper(df, zero(T), l, u)
    # Use the barrier-aware preconditioner to define
    # barrier-aware optimization method instance (precondition relevance)
    _optimizer = barrier_method(F.method, P, (P, x) -> F.precondprep(P, x, l, u, dfbox))

    # we wait until state has been initialized to set the initial mu because we need the gradient of the objective
    state = initial_state(_optimizer, options, df, x)
    @assert state.x == x
    f_x = state.f_x
    g_x = if hasproperty(state, :g_x)
        copy(state.g_x)
    else
        copy(gradient!(dfbox, x))
    end
    box = BoxBarrier(l, u)
    mu = initial_mu(box, x, g_x, F)

    if show_trace > 00
        println("Fminbox")
        println("-------")
        print("Initial mu = ")
        show(IOContext(stdout, :compact => true), "text/plain", dfbox.mu)
        println("\n")
    end

    # First iteration
    iteration = 1
    _time = time()

    # Optimize with current setting of mu
    if show_trace > 0
        header_string = "Fminbox iteration $iteration"
        println(header_string)
        println("-"^length(header_string))
        print("Calling inner optimizer with mu = ")
        show(IOContext(stdout, :compact => true), "text/plain", dfbox.mu)
        println("\n")
        println("(numbers below include barrier contribution)")
    end

    # We add the barrier term to the objective function
    # Since this changes the objective of the inner optimizer, we have to reset its state
    dfbox = BarrierWrapper(df, mu, l, u)
    reset!(_optimizer, state, dfbox, x)

    # Store current state
    x_previous = copy(x)
    f_x_previous = f_x

    results = optimize(dfbox, x, _optimizer, options, state)
    stopped_by_callback = results.stopped_by.callback
    dfbox.obj.f_calls[1] = 0
    if hasfield(typeof(dfbox.obj), :df_calls)
        dfbox.obj.df_calls[1] = 0
    end
    if hasfield(typeof(dfbox.obj), :h_calls)
        dfbox.obj.h_calls[1] = 0
    end

    # Compute function value and gradient (without barrier term)
    copyto!(x, minimizer(results))
    f_x, _g_x = value_gradient!(df, x)
    copyto!(g_x, _g_x)

    boxdist = Base.minimum(((xi, li, ui),) -> min(xi - li, ui - xi), zip(x, l, u)) # Base.minimum !== minimum
    if show_trace > 0
        println()
        println("Exiting inner optimizer with x = ", x)
        print("Current distance to box: ")
        show(IOContext(stdout, :compact => true), "text/plain", boxdist)
        println()
        println("Decreasing barrier term μ.\n")
    end

    # Test for convergence
    g = x .- clamp.(x .- g_x, l, u)
    _x_converged, _f_converged, _g_converged, f_increased =
        assess_convergence(
            x,
            x_previous,
            f_x,
            f_x_previous,
            g,
            options.outer_x_abstol,
            options.outer_x_reltol,
            options.outer_f_abstol,
            options.outer_f_reltol,
            options.outer_g_abstol,
        )
    converged =
        _x_converged ||
        _f_converged ||
        _g_converged ||
        stopped_by_callback
    _time = time()
    stopped_by_time_limit = _time - t0 > options.time_limit
    stopped = stopped_by_time_limit

    if f_increased && !allow_outer_f_increases
        @warn("f(x) increased: stopping optimization")
    else
        while !converged && !stopped && iteration < outer_iterations
            # Increment the number of steps we've had to perform
            iteration += 1

            # Decrease mu
            mu *= T(F.mufactor)

            # Optimize with current setting of mu
            if show_trace > 0
                header_string = "Fminbox iteration $iteration"
                println(header_string)
                println("-"^length(header_string))
                print("Calling inner optimizer with mu = ")
                show(IOContext(stdout, :compact => true), "text/plain", dfbox.mu)
                println("\n")
                println("(numbers below include barrier contribution)")
            end

            # We need to update the barrier term of the objective function.
            # Since this changes the objective of the inner optimizer, we have to reset its state
            dfbox = BarrierWrapper(df, mu, l, u)
            reset!(_optimizer, state, dfbox, x)

            # Store current state
            copyto!(x_previous, x)
            f_x_previous = f_x

            resultsnew = optimize(dfbox, x, _optimizer, options, state)
            stopped_by_callback = resultsnew.stopped_by.callback
            append!(results, resultsnew)
            dfbox.obj.f_calls[1] = 0
            if hasfield(typeof(dfbox.obj), :df_calls)
                dfbox.obj.df_calls[1] = 0
            end
            if hasfield(typeof(dfbox.obj), :h_calls)
                dfbox.obj.h_calls[1] = 0
            end

            # Compute function value and gradient (without barrier term)
            copyto!(x, minimizer(results))
            f_x, _g_x = value_gradient!(df, x)
            copyto!(g_x, _g_x)

            boxdist = Base.minimum(((xi, li, ui),) -> min(xi - li, ui - xi), zip(x, l, u)) # Base.minimum !== minimum
            if show_trace > 0
                println()
                println("Exiting inner optimizer with x = ", x)
                print("Current distance to box: ")
                show(IOContext(stdout, :compact => true), "text/plain", boxdist)
                println()
                println("Decreasing barrier term μ.\n")
            end

            # Test for convergence
            g .= x .- clamp.(g_x, l, u)
            _x_converged, _f_converged, _g_converged, f_increased =
                assess_convergence(
                    x,
                    x_previous,
                    f_x,
                    f_x_previous,
                    g,
                    options.outer_x_abstol,
                    options.outer_x_reltol,
                    options.outer_f_abstol,
                    options.outer_f_reltol,
                    options.outer_g_abstol,
                )
            converged =
                _x_converged ||
                _f_converged ||
                _g_converged ||
                stopped_by_callback
            if f_increased && !allow_outer_f_increases
                @warn("f(x) increased: stopping optimization")
                break
            end
            _time = time()
            stopped_by_time_limit = _time - t0 > options.time_limit
            stopped = stopped_by_time_limit
        end
    end

    stopped_by = (
        f_limit_reached = false,
        g_limit_reached = false,
        h_limit_reached = false,
        time_limit = stopped_by_time_limit,
        callback = stopped_by_callback,
        f_increased = f_increased && !options.allow_f_increases,
        ls_failed = false,
        iterations = results.stopped_by.iterations,
        x_converged = _x_converged,
        f_converged = _f_converged,
        g_converged = _g_converged,
    )
    termination_code = _termination_code(df, g_residual(g), BoxState(x, f_x, x_previous, f_x_previous), stopped_by, options)

    return MultivariateOptimizationResults(
        F,
        initial_x,
        x,
        f_x,
        iteration,
        results.x_abstol,
        results.x_reltol,
        x_abschange(x, x_previous),
        x_relchange(x, x_previous),
        results.f_abstol,
        results.f_reltol,
        f_abschange(f_x, f_x_previous),
        f_relchange(f_x, f_x_previous),
        results.g_abstol,
        g_residual(g),
        results.trace,
        results.f_calls,
        results.g_calls,
        results.h_calls,
        options.time_limit,
        _time - t0,
        stopped_by,
        termination_code,
    )
end
