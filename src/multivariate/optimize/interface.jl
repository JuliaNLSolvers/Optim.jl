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
promote_objtype(method, initial_x, obj_args...) = error("No default objective type for $method and $obj_args.")
# actual promotions, notice that (args...) captures FirstOrderOptimizer and NonDifferentiable, etc
promote_objtype(method::ZerothOrderOptimizer, x, ::Val{S}, args...) where S = NonDifferentiable(args..., x, real(zero(eltype(x))), Val{S}())
promote_objtype(method::FirstOrderOptimizer,  x, ::Val{S}, args...) where S = OnceDifferentiable(args..., x, real(zero(eltype(x))), Val{:central}(), Val{S}())
promote_objtype(method::FirstOrderOptimizer,  x, ::Val{S}, f, g!, h!) where S = OnceDifferentiable(f, g!, x, real(zero(eltype(x))), Val{:central}(), Val{S}())
promote_objtype(method::SecondOrderOptimizer, x, ::Val{S}, args...) where S = TwiceDifferentiable(args..., x, real(zero(eltype(x))), Val{:central}(), Val{S}())
# no-op
promote_objtype(method::ZerothOrderOptimizer, x, ::Val, nd::NonDifferentiable)  = nd
promote_objtype(method::ZerothOrderOptimizer, x, ::Val, od::OnceDifferentiable) = od
promote_objtype(method::FirstOrderOptimizer,  x, ::Val, od::OnceDifferentiable) = od
promote_objtype(method::ZerothOrderOptimizer, x, ::Val, td::TwiceDifferentiable) = td
promote_objtype(method::FirstOrderOptimizer,  x, ::Val, td::TwiceDifferentiable) = td
promote_objtype(method::SecondOrderOptimizer, x, ::Val, td::TwiceDifferentiable) = td


# if on method or options are present
optimize(f,         initial_x::AbstractArray; kwargs...) = optimize((f,),        initial_x; kwargs...)
optimize(f, g!,     initial_x::AbstractArray; kwargs...) = optimize((f, g!),     initial_x; kwargs...)
optimize(f, g!, h!, initial_x::AbstractArray; kwargs...) = optimize((f, g!, h!), initial_x; kwargs...)
function optimize(f::Tuple, initial_x::AbstractArray; kwargs...)
    method = fallback_method(f...)
    d = promote_objtype(method, initial_x, f...)
    checked_kwargs, method = check_kwargs(kwargs, method)
    add_default_opts!(checked_kwargs, method)

    options = Options(; checked_kwargs...)
    optimize(d, initial_x, method, options)
end

# no method supplied
optimize(d::T,      initial_x::AbstractArray, options::Options) where T<:AbstractObjective = optimize((d,), initial_x, fallback_method(T), options)
optimize(f,         initial_x::AbstractArray, options::Options) = optimize((f,),        initial_x, fallback_method(f), options)
optimize(f, g!,     initial_x::AbstractArray, options::Options) = optimize((f, g!),     initial_x, fallback_method(f), options)
optimize(f, g!, h!, initial_x::AbstractArray, options::Options) = optimize((f, g!, h!), initial_x, fallback_method(f), options)

# potentially everything is supplied (besides caches)
optimize(f,         initial_x::AbstractArray, method::AbstractOptimizer,
         options::Options = InternalUseOptions(method)) = optimize((f,),        initial_x, method, options)
optimize(f, g!,     initial_x::AbstractArray, method::AbstractOptimizer,
         options::Options = InternalUseOptions(method)) = optimize((f, g!),     initial_x, method, options)
optimize(f, g!, h!, initial_x::AbstractArray, method::AbstractOptimizer,
         options::Options = InternalUseOptions(method)) = optimize((f, g!, h!), initial_x, method, options)
@inline function optimize(f::Tuple, initial_x::AbstractArray, method::AbstractOptimizer,
                  options::Options{T, TCallback, S} = InternalUseOptions(method)) where {T, TCallback, S}
    d = promote_objtype(method, initial_x, Val{S}(), f...)

    optimize(d, initial_x, method, options)
end

function optimize(d::D, initial_x::AbstractArray, method::SecondOrderOptimizer,
                  options::Options = InternalUseOptions(method)) where {D <: Union{NonDifferentiable, OnceDifferentiable}}
    d = promote_objtype(method, initial_x, d)
    optimize(d, initial_x, method, options)
end
