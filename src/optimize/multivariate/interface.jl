const ZerothOrderSolver = Union{NelderMead, SimulatedAnnealing, ParticleSwarm}
const FirstOrderSolver = Union{AcceleratedGradientDescent, ConjugateGradient, GradientDescent,
                               MomentumGradientDescent, BFGS, LBFGS}
const SecondOrderSolver = Union{Newton, NewtonTrustRegion}
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

fallback_method(d::AbstractObjective) = err("$d has no fallback method")
fallback_method(d::NonDifferentiable) = NelderMead()
fallback_method(f) = NelderMead()
fallback_method(d::OnceDifferentiable) = LBFGS()
fallback_method(f, g!) = LBFGS()
fallback_method(d::TwiceDifferentiable) = Newton()
fallback_method(f, g!, h!) = Newton()

promote_objtype(method, initial_x, obj_args...) = err("No default objective type for $method and $obj_args.")
promote_objtype(method::ZerothOrderSolver, initial_x, obj_args...) = NonDifferentiable(obj_args..., initial_x)
promote_objtype(method::FirstOrderSolver,  initial_x, obj_args...) = OnceDifferentiable(obj_args..., initial_x)
promote_objtype(method::SecondOrderSolver, initial_x, obj_args...) = TwiceDifferentiable(obj_args..., initial_x)

# For each "objective" input (callables or AbstractObjectives) we need to define
# four versions: no method with kwargs for options (I), method with kwargs for options (II),
# method with Options (III), and Options without method (IV)

## AbstractObjectives
# no method + (potentially) without initial_x + kwargs
# use objective.last_f_x, since this is guaranteed to be in Non-, Once-, and TwiceDifferentiable
function optimize(objective::AbstractObjective, initial_x::AbstractArray = objective.last_x_f; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, fallback_method(objective))
    optimize(objective, initial_x, method, Options(; checked_kwargs...))
end
# no method + no initial_x + Options
optimize(d::AbstractObjective,                           options::Options) = optimize(d, d.last_x_f, fallback_method(d), options)
# no method + Options
optimize(d::AbstractObjective, initial_x::AbstractArray, options::Options) = optimize(d, initial_x,  fallback_method(d), options)

## f only - these methods define the behavior when only the objective function is passed
function optimize(f, initial_x::AbstractArray; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, fallback_method(f))
    d = promote_objtype(method, initial_x, f)
    optimize(d, initial_x, method, Options(; checked_kwargs...))
end
function optimize(f, initial_x::AbstractArray, options::Options)
    method = fallback_method(f)
    d = promote_objtype(method, initial_x, f)
    optimize(d, initial_x, method, options, initial_state(method, options, d, initial_x))
end
function optimize{M <: Union{Optimizer, Void}}(f, initial_x::AbstractArray,
                                  method::M, options::Options = Options())
    d = promote_objtype(method, initial_x, f)
    optimize(d, initial_x, method, options, initial_state(method, options, d, initial_x))
end


## f and g! - these methods define the behavior when the objective function and gradient is passed
function optimize(f, g!, initial_x::AbstractArray; kwargs...)
    method = fallback_method(f, g!)
    d = promote_objtype(method, initial_x, f, g!)
    checked_kwargs, method = check_kwargs(kwargs, method)
    options = Options(; checked_kwargs...)
    optimize(d, initial_x, method, options, initial_state(method, options, d, initial_x))
end
function optimize(f, g!, initial_x::AbstractArray, options::Options)
    d = OnceDifferentiable(f, g!, initial_x)
    optimize(d, initial_x, fallback_method(d), options)
end
function optimize{M<:Optimizer}(f, g!, initial_x::AbstractArray, method::M, options::Options = Options())
    d = promote_objtype(method, initial_x, f, g!)
    optimize(d, initial_x, method, options, initial_state(method, options, d, initial_x))
end

## f, g!, and h!
function optimize(f, g!, h!, initial_x::AbstractArray; kwargs...)
    d = TwiceDifferentiable(f, g!, h!, initial_x)
    method = fallback_method(d)
    checked_kwargs, method = check_kwargs(kwargs, method)
    options = Options(; checked_kwargs...)
    optimize(d, initial_x, method, options, initial_state(method, options, d, initial_x))
end
function optimize(f, g!, h!, initial_x::AbstractArray, options::Options)
    if isa(method, ZerothOrderSolver)
        d = NonDifferentiable(f, initial_x)
    elseif isa(method, FirstOrderSolver)
        d = OnceDifferentiable(f, g!, initial_x)
    elseif isa(method, SecondOrderSolver)
        d = TwiceDifferentiable(f, g!, h!, initial_x)
    end
    method = fallback_method(d)
    optimize(d, initial_x, method, options, initial_state(method, options, d, initial_x))
end
function optimize(f, g!, h!, initial_x::AbstractArray, method::Optimizer, options::Options = Options())
    if isa(method, ZerothOrderSolver)
        d = NonDifferentiable(f, initial_x)
    elseif isa(method, FirstOrderSolver)
        d = OnceDifferentiable(f, g!, initial_x)
    elseif isa(method, SecondOrderSolver)
        d = TwiceDifferentiable(f, g!, h!, initial_x)
    end
    optimize(d, initial_x, method, options, initial_state(method, options, d, initial_x))
end

# OnceDifferentiable input for automatic promotion using autodiff
function optimize{D <: Union{NonDifferentiable, OnceDifferentiable}}(d::D, initial_x::AbstractArray, method::SecondOrderSolver, options::Options = Options())
    d = promote_objtype(method, initial_x, d)
    optimize(d, initial_x, method, options, initial_state(method, options, d, initial_x))
end
