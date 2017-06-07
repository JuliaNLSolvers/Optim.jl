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

fallback_method(d::NonDifferentiable) = NelderMead()
fallback_method(d::OnceDifferentiable) = LBFGS()
fallback_method(d::TwiceDifferentiable) = Newton()
fallback_method(d) = NelderMead()
fallback_method(f, g!) = LBFGS()
fallback_method(f, g!, h!) = Newton()

promote_objtype(method, initial_x, obj_args...) = error("No default objective type for $method and $obj_args.")
promote_objtype(method::ZerothOrderSolver, initial_x, obj_args...) = NonDifferentiable(obj_args..., initial_x)
promote_objtype(method::ZerothOrderSolver, initial_x, od::OnceDifferentiable) = od
promote_objtype(method::ZerothOrderSolver, initial_x, td::TwiceDifferentiable) = td
promote_objtype(method::ZerothOrderSolver, initial_x, nd::NonDifferentiable) = nd
promote_objtype(method::FirstOrderSolver,  initial_x, obj_args...) = OnceDifferentiable(obj_args..., initial_x)
promote_objtype(method::FirstOrderSolver,  initial_x, od::OnceDifferentiable) = od
promote_objtype(method::FirstOrderSolver,  initial_x, td::TwiceDifferentiable) = td
promote_objtype(method::FirstOrderSolver,  initial_x, f, g!, h!)   = OnceDifferentiable(f, g!, initial_x)
promote_objtype(method::SecondOrderSolver, initial_x, obj_args...) = TwiceDifferentiable(obj_args..., initial_x)
promote_objtype(method::SecondOrderSolver, initial_x, td::TwiceDifferentiable) = td

# use objective.last_f_x, since this is guaranteed to be in Non-, Once-, and TwiceDifferentiable
function optimize(objective::AbstractObjective, initial_x::AbstractArray = objective.last_x_f; kwargs...)
    method = fallback_method(objective)
    checked_kwargs, method = check_kwargs(kwargs, method)
    options = Options(; checked_kwargs...)
    optimize(objective, initial_x, method, options)
end

optimize(d::AbstractObjective,                           options::Options) = optimize(d, d.last_x_f, fallback_method(d), options)
optimize(d::AbstractObjective, method::Optimizer,        options::Options = Options()) = optimize(d, d.last_x_f, method, options)
optimize(d::AbstractObjective, initial_x::AbstractArray, options::Options) = optimize(d, initial_x,  fallback_method(d), options)

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

optimize(f,         initial_x::AbstractArray, options::Options) = optimize((f,),        initial_x, options)
optimize(f, g!,     initial_x::AbstractArray, options::Options) = optimize((f, g!),     initial_x, options)
optimize(f, g!, h!, initial_x::AbstractArray, options::Options) = optimize((f, g!, h!), initial_x, options)
function optimize(f::Tuple, initial_x::AbstractArray, options::Options)
    method = fallback_method(f...)
    d = promote_objtype(method, initial_x, f...)
    optimize(d, initial_x, method, options)
end

optimize(f,         initial_x::AbstractArray, method::Optimizer, options::Options = Options()) = optimize((f,),        initial_x, method, options)
optimize(f, g!,     initial_x::AbstractArray, method::Optimizer, options::Options = Options()) = optimize((f, g!),     initial_x, method, options)
optimize(f, g!, h!, initial_x::AbstractArray, method::Optimizer, options::Options = Options()) = optimize((f, g!, h!), initial_x, method, options)
function optimize(f::Tuple, initial_x::AbstractArray, method::Optimizer, options::Options = Options())
    d = promote_objtype(method, initial_x, f...)
    optimize(d, initial_x, method, options)
end

function optimize{D <: Union{NonDifferentiable, OnceDifferentiable}}(d::D, initial_x::AbstractArray, method::SecondOrderSolver, options::Options = Options())
    d = promote_objtype(method, initial_x, d)
    optimize(d, initial_x, method, options)
end

initialize_objective(d::UninitializedNonDifferentiable, x) = NonDifferentiable(d, x)
initialize_objective(d::UninitializedOnceDifferentiable, x) = OnceDifferentiable(d, x)
initialize_objective(d::UninitializedTwiceDifferentiable, x) = TwiceDifferentiable(d, x)
function optimize(d::UninitializedObjective, initial_x::AbstractArray, method::Optimizer, options::Options = Options())
    id = initialize_objective(d, initial_x)
    id = promote_objtype(method, initial_x, id)
    optimize(id, initial_x, method, options, initial_state(method, options, id, initial_x))
end
function optimize(d::UninitializedObjective, initial_x::AbstractArray)
    id = initialize_objective(d, initial_x)
    optimize(id)
end
