Base.@deprecate method(x) summary(x)

const has_deprecated_fminbox = Ref(false)
function optimize(
        df::OnceDifferentiable,
        initial_x::Array{T},
        l::Array{T},
        u::Array{T},
        ::Type{Fminbox};
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
        linesearch = LineSearches.HagerZhang{T}(),
        eta::Real = convert(T,0.4),
        mu0::T = convert(T, NaN),
        mufactor::T = convert(T, 0.001),
        precondprep = (P, x, l, u, mu) -> precondprepbox!(P, x, l, u, mu),
        optimizer = ConjugateGradient,
        optimizer_o = Options(store_trace = store_trace,
                                          show_trace = show_trace,
                                          extended_trace = extended_trace),
        nargs...) where T<:AbstractFloat
        if !has_deprecated_fminbox[]
            @warn("Fminbox with the optimizer keyword is deprecated, construct Fminbox{optimizer}() and pass it to optimize(...) instead.")
            has_deprecated_fminbox[] = true
        end
        optimize(df, initial_x, l, u, Fminbox{optimizer}();
                 allow_f_increases=allow_f_increases,
                 iterations=iterations,
                 store_trace=store_trace,
                 show_trace=show_trace,
                 extended_trace=extended_trace,
                 show_every=show_every,
                 callback=callback,
                 linesearch=linesearch,
                 eta=eta,
                 mu0=mu0,
                 mufactor=mufactor,
                 precondprep=precondprep,
                 optimizer_o=optimizer_o)
end

function optimize(::AbstractObjective)
    throw(ErrorException("Optimizing an objective `obj` without providing an initial `x` has been deprecated without backwards compatability. Please explicitly provide an `x`: `optimize(obj, x)``"))
end
function optimize(::AbstractObjective, ::Method)
    throw(ErrorException("Optimizing an objective `obj` without providing an initial `x` has been deprecated without backwards compatability. Please explicitly provide an `x`: `optimize(obj, x, method)``"))
end
function optimize(::AbstractObjective, ::Method, ::Options)
    throw(ErrorException("Optimizing an objective `obj` without providing an initial `x` has been deprecated without backwards compatability. Please explicitly provide an `x`: `optimize(obj, x, method, options)``"))
end
function optimize(::AbstractObjective, ::Options)
    throw(ErrorException("Optimizing an objective `obj` without providing an initial `x` has been deprecated without backwards compatability. Please explicitly provide an `x`: `optimize(obj, x, options)``"))
end

function optimize(df::OnceDifferentiable,
    l::Array{T},
    u::Array{T},
    F::Fminbox{O}; kwargs...) where {T<:AbstractFloat,O<:AbstractOptimizer}
    throw(ErrorException("Optimizing an objective `obj` without providing an initial `x` has been deprecated without backwards compatability. Please explicitly provide an `x`: `optimize(obj, x, l, u, method, options)``"))
end
