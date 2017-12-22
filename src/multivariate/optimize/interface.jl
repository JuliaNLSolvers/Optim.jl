const ZerothOrderSolver = Union{NelderMead, SimulatedAnnealing, ParticleSwarm}
const FirstOrderSolver = Union{AcceleratedGradientDescent, ConjugateGradient, GradientDescent,
                               MomentumGradientDescent, BFGS, LBFGS}
const SecondOrderSolver = Union{Newton, NewtonTrustRegion, KrylovTrustRegion}
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

fallback_method(f) = NelderMead()
fallback_method(f, g!) = LBFGS()
fallback_method(f, g!, h!) = Newton()

fallback_method(d::OnceDifferentiable) = LBFGS()
fallback_method(d::TwiceDifferentiable) = Newton()

# promote the objective (tuple of callables or an AbstractObjective) according to method requirement
promote_objtype(method, initial_x, obj_args...) = error("No default objective type for $method and $obj_args.")
# actual promotions, notice that (args...) captures FirstOrderSolver and NonDifferentiable, etc
promote_objtype(method::ZerothOrderSolver, x, args...) = NonDifferentiable(args..., x, real(zero(eltype(x))))
promote_objtype(method::FirstOrderSolver,  x, args...) = OnceDifferentiable(args..., x, real(zero(eltype(x))))
promote_objtype(method::FirstOrderSolver,  x, f, g!, h!) = OnceDifferentiable(f, g!, x, real(zero(eltype(x))))
promote_objtype(method::SecondOrderSolver, x, args...) = TwiceDifferentiable(args..., x, real(zero(eltype(x))))
# no-op
promote_objtype(method::ZerothOrderSolver, x, nd::NonDifferentiable)  = nd
promote_objtype(method::ZerothOrderSolver, x, od::OnceDifferentiable) = od
promote_objtype(method::FirstOrderSolver,  x, od::OnceDifferentiable) = od
promote_objtype(method::ZerothOrderSolver, x, td::TwiceDifferentiable) = td
promote_objtype(method::FirstOrderSolver,  x, td::TwiceDifferentiable) = td
promote_objtype(method::SecondOrderSolver, x, td::TwiceDifferentiable) = td

# if on method or options are present
optimize(f,         initial_x::AbstractArray; kwargs...) = optimize((f,),        initial_x; kwargs...)
optimize(f, g!,     initial_x::AbstractArray; kwargs...) = optimize((f, g!),     initial_x; kwargs...)
optimize(f, g!, h!, initial_x::AbstractArray; kwargs...) = optimize((f, g!, h!), initial_x; kwargs...)
function optimize(f::Tuple, initial_x::AbstractArray; kwargs...)
    method = fallback_method(f...)
    d = promote_objtype(method, initial_x, f...)
    checked_kwargs, method = check_kwargs(kwargs, method)
    options = Options(; checked_kwargs...)
    optimize(d, initial_x, method, options)
end

# no method supplied
optimize(d::T,      initial_x::AbstractArray, options::Options) where T<:AbstractObjective = optimize((d,), initial_x, fallback_method(T), options)
optimize(f,         initial_x::AbstractArray, options::Options) = optimize((f,),        initial_x, fallback_method(f), options)
optimize(f, g!,     initial_x::AbstractArray, options::Options) = optimize((f, g!),     initial_x, fallback_method(f), options)
optimize(f, g!, h!, initial_x::AbstractArray, options::Options) = optimize((f, g!, h!), initial_x, fallback_method(f), options)

# potentially everything is supplied (besides caches)
optimize(f,         initial_x::AbstractArray, method::Optimizer, options::Options = Options()) = optimize((f,),        initial_x, method, options)
optimize(f, g!,     initial_x::AbstractArray, method::Optimizer, options::Options = Options()) = optimize((f, g!),     initial_x, method, options)
optimize(f, g!, h!, initial_x::AbstractArray, method::Optimizer, options::Options = Options()) = optimize((f, g!, h!), initial_x, method, options)
function optimize(f::Tuple, initial_x::AbstractArray, method::Optimizer, options::Options = Options())
    d = promote_objtype(method, initial_x, f...)

    optimize(d, initial_x, method, options)
end

function optimize(d::D, initial_x::AbstractArray, method::SecondOrderSolver, options::Options = Options()) where {D <: Union{NonDifferentiable, OnceDifferentiable}}
    d = promote_objtype(method, initial_x, d)
    optimize(d, initial_x, method, options)
end
