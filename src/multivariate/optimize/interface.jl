# Multivariate optimization
function check_kwargs(kwargs, fallback_method)
    kws = Dict{Symbol, Any}()
    method = nothing
    for kwarg in kwargs
        if kwarg[1] != :method
            kws[kwarg[1]] = kwarg[2]
        else
            method = kwarg[2]
        end
    end

    if method == nothing
        method = fallback_method
    end
    kws, method
end

default_options(method::AbstractOptimizer) = Dict{Symbol, Any}()

function add_default_opts!(opts::Dict{Symbol, Any}, method::AbstractOptimizer)
    for newopt in default_options(method)
        if !haskey(opts, newopt[1])
            opts[newopt[1]] = newopt[2]
        end
    end
end

fallback_method(f) = NelderMead()
fallback_method(f, g!) = LBFGS()
fallback_method(f, g!, h!) = Newton()

fallback_method(d::OnceDifferentiable) = LBFGS()
fallback_method(d::TwiceDifferentiable) = Newton()

# promote the objective (tuple of callables or an AbstractObjective) according to method requirement
promote_objtype(method, initial_x, autodiff::Symbol, inplace::Bool, args...) = error("No default objective type for $method and $args.")
# actual promotions, notice that (args...) captures FirstOrderOptimizer and NonDifferentiable, etc
promote_objtype(method::ZerothOrderOptimizer, x, autodiff::Symbol, inplace::Bool, args...) = NonDifferentiable(args..., x, real(zero(eltype(x))))
promote_objtype(method::FirstOrderOptimizer,  x, autodiff::Symbol, inplace::Bool, f) = OnceDifferentiable(f, x, real(zero(eltype(x))); autodiff = autodiff)
promote_objtype(method::FirstOrderOptimizer,  x, autodiff::Symbol, inplace::Bool, args...) = OnceDifferentiable(args..., x, real(zero(eltype(x))); inplace = inplace)
promote_objtype(method::FirstOrderOptimizer,  x, autodiff::Symbol, inplace::Bool, f, g, h) = OnceDifferentiable(f, g, x, real(zero(eltype(x))); inplace = inplace)
promote_objtype(method::SecondOrderOptimizer, x, autodiff::Symbol, inplace::Bool, f) = TwiceDifferentiable(f, x, real(zero(eltype(x))); autodiff = autodiff)
promote_objtype(method::SecondOrderOptimizer, x, autodiff::Symbol, inplace::Bool, f::NotInplaceObjective) = TwiceDifferentiable(f, x, real(zero(eltype(x))))
promote_objtype(method::SecondOrderOptimizer, x, autodiff::Symbol, inplace::Bool, f::InplaceObjective) = TwiceDifferentiable(f, x, real(zero(eltype(x))))
promote_objtype(method::SecondOrderOptimizer, x, autodiff::Symbol, inplace::Bool, f, g) = TwiceDifferentiable(f, g, x, real(zero(eltype(x))); inplace = inplace, autodiff = autodiff)
promote_objtype(method::SecondOrderOptimizer, x, autodiff::Symbol, inplace::Bool, f, g, h) = TwiceDifferentiable(f, g, h, x, real(zero(eltype(x))); inplace = inplace)
# no-op
promote_objtype(method::ZerothOrderOptimizer, x, autodiff::Symbol, inplace::Bool, nd::NonDifferentiable)  = nd
promote_objtype(method::ZerothOrderOptimizer, x, autodiff::Symbol, inplace::Bool, od::OnceDifferentiable) = od
promote_objtype(method::FirstOrderOptimizer,  x, autodiff::Symbol, inplace::Bool, od::OnceDifferentiable) = od
promote_objtype(method::ZerothOrderOptimizer, x, autodiff::Symbol, inplace::Bool, td::TwiceDifferentiable) = td
promote_objtype(method::FirstOrderOptimizer,  x, autodiff::Symbol, inplace::Bool, td::TwiceDifferentiable) = td
promote_objtype(method::SecondOrderOptimizer, x, autodiff::Symbol, inplace::Bool, td::TwiceDifferentiable) = td

# if no method or options are present
function optimize(f,         initial_x::AbstractArray; inplace = true, autodiff = :finite, kwargs...)
    method = fallback_method(f)
    checked_kwargs, method = check_kwargs(kwargs, method)
    d = promote_objtype(method, initial_x, autodiff, inplace, f)
    add_default_opts!(checked_kwargs, method)

    options = Options(; checked_kwargs...)
    optimize(d, initial_x, method, options)
end
function optimize(f, g, initial_x::AbstractArray; inplace = true, autodiff = :finite, kwargs...)

    method = fallback_method(f, g)
    checked_kwargs, method = check_kwargs(kwargs, method)
    d = promote_objtype(method, initial_x, autodiff, inplace, f, g)
    add_default_opts!(checked_kwargs, method)

    options = Options(; checked_kwargs...)
    optimize(d, initial_x, method, options)
end
function optimize(f, g, h, initial_x::AbstractArray; inplace = true, autodiff = :finite, kwargs...)

    method = fallback_method(f, g, h)
    checked_kwargs, method = check_kwargs(kwargs, method)
    d = promote_objtype(method, initial_x, autodiff, inplace, f, g, h)
    add_default_opts!(checked_kwargs, method)

    options = Options(; checked_kwargs...)
    optimize(d, initial_x, method, options)
end

# no method supplied with objective
function optimize(d::T, initial_x::AbstractArray, options::Options) where T<:AbstractObjective
    optimize(d, initial_x, fallback_method(d), options)
end
# no method supplied with inplace and autodiff keywords becauase objective is not supplied
function optimize(f, initial_x::AbstractArray, options::Options; inplace = true, autodiff = :finite)
    method = fallback_method(f)
    d = promote_objtype(method, initial_x, autodiff, inplace, f)
    optimize(d, initial_x, method, options)
end
function optimize(f, g, initial_x::AbstractArray, options::Options; inplace = true, autodiff = :finite)

    method = fallback_method(f, g)
    d = promote_objtype(method, initial_x, autodiff, inplace, f, g)
    optimize(d, initial_x, method, options)
end
function optimize(f, g, h, initial_x::AbstractArray{T}, options::Options; inplace = true, autodiff = :finite) where {T}

    method = fallback_method(f, g, h)
    d = promote_objtype(method, initial_x, autodiff, inplace, f, g, h)

    optimize(d, initial_x, method, options)
end

# potentially everything is supplied (besides caches)
function optimize(f, initial_x::AbstractArray, method::AbstractOptimizer,
                     options::Options = Options(;default_options(method)...); inplace = true, autodiff = :finite)

    d = promote_objtype(method, initial_x, autodiff, inplace, f)
    optimize(d, initial_x, method, options)
end
function optimize(f, g, initial_x::AbstractArray, method::AbstractOptimizer,
         options::Options = Options(;default_options(method)...); inplace = true, autodiff = :finite)

    d = promote_objtype(method, initial_x, autodiff, inplace, f, g)

    optimize(d, initial_x, method, options)
end
function optimize(f, g, h, initial_x::AbstractArray{T}, method::AbstractOptimizer,
         options::Options = Options(;default_options(method)...); inplace = true, autodiff = :finite) where T

    d = promote_objtype(method, initial_x, autodiff, inplace, f, g, h)

    optimize(d, initial_x, method, options)
end

function optimize(d::D, initial_x::AbstractArray, method::SecondOrderOptimizer,
                  options::Options = Options(;default_options(method)...); autodiff = :finite, inplace = true) where {D <: Union{NonDifferentiable, OnceDifferentiable}}
    d = promote_objtype(method, initial_x, autodiff, inplace, d)
    optimize(d, initial_x, method, options)
end
