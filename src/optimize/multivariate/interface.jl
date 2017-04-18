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
fallback_method(d::OnceDifferentiable) = LBFGS()
fallback_method(d::TwiceDifferentiable) = Newton()

## AbstractObjectives
# no method + (potentially) without initial_x + kwargs
# use objective.last_f_x, since this is guaranteed to be in Non-, Once-, and TwiceDifferentiable
function optimize(objective::AbstractObjective, initial_x::AbstractArray = objective.last_x_f; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, fallback_method(objective))
    optimize(objective, initial_x, method, Options(; checked_kwargs...))
end
# no method + Options
optimize(d::AbstractObjective, initial_x::AbstractArray, options::Options) = optimize(d, initial_x,  fallback_method(d), options)
# no method + no initial_x + Options
optimize(d::AbstractObjective,                           options::Options) = optimize(d, d.last_x_f, fallback_method(d), options)

# f only - these methods define the behavior when only the objective function is passed
function optimize(f::Function, initial_x::Array; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, NelderMead())
    optimize(f, initial_x, method, Options(; checked_kwargs...))
end

function optimize{M <: Union{FirstOrderSolver, SecondOrderSolver}}(f, initial_x::AbstractArray, method::M, options::Options)
    d = ifelse(M <: FirstOrderSolver, OnceDifferentiable(f, initial_x), TwiceDifferentiable(f, initial_x))
    optimize(d, initial_x, method, options)
end
function optimize{M <: Union{FirstOrderSolver, SecondOrderSolver}}(f::Function, initial_x::AbstractArray, method::M, options::Options)
    d = ifelse(M <: FirstOrderSolver, OnceDifferentiable(f, initial_x), TwiceDifferentiable(f, initial_x))
    optimize(d, initial_x, method, options)
end

optimize(f, initial_x::AbstractArray, method::Optimizer, options::Options = Options()) = optimize(NonDifferentiable(f, initial_x), initial_x, method,       options)
optimize(f, initial_x::AbstractArray,                    options::Options)             = optimize(NonDifferentiable(f, initial_x), initial_x, NelderMead(), options)

optimize(f::Function, initial_x::AbstractArray, method::Optimizer, options::Options = Options()) = optimize(NonDifferentiable(f, initial_x), initial_x, method,       options)
optimize(f::Function, initial_x::AbstractArray,                    options::Options) = optimize(NonDifferentiable(f, initial_x), initial_x, NelderMead(), options)

# f and g! - these methods define the behavior when the objective function and gradient is passed
function optimize(f::Function, g!::Function, initial_x::Array; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, BFGS())
    optimize(f, g!, initial_x, method, Options(; checked_kwargs...))
end
function optimize(f, g!, initial_x::AbstractArray, options::Options = Options())
    d = OnceDifferentiable(f, g!, initial_x)
    optimize(d, initial_x, fallback_method(d), options)
end
function optimize(f, g!, initial_x::AbstractArray, method::Optimizer, options::Options = Options())
    d = OnceDifferentiable(f, g!, initial_x)
    optimize(d, initial_x, method, options)
end
# (abstract) functions
function optimize(f::Function, g!::Function, initial_x::AbstractArray, options::Options = Options())
    d = OnceDifferentiable(f, g!, initial_x)
    optimize(d, initial_x, fallback_method(d), options)
end
function optimize(f::Function, g!::Function, initial_x::AbstractArray, method::Optimizer, options::Options = Options())
    d = OnceDifferentiable(f, g!, initial_x)
    optimize(d, initial_x, method, options)
end

## f, g!, and h!
# no method + kwargs
function optimize(f::Function, g!::Function, h!::Function, initial_x::Array; kwargs...)
    checked_kwargs, method = check_kwargs(kwargs, Newton())
    optimize(f, g!, h!, initial_x, method, Options(; checked_kwargs...))
end
# no method + options
function optimize(f, g!, h!, initial_x::AbstractArray, options)
    d = TwiceDifferentiable(f, g!, h!, initial_x)
    optimize(d, initial_x, fallback_method(d), options)
end
# method + Options (or default value if no Options passed)
function optimize(f, g!, h!, initial_x::AbstractArray, method::Optimizer, options::Options = Options())
    d = TwiceDifferentiable(f, g!, h!, initial_x)
    optimize(d, initial_x, method, options)
end
# no method + Options
function optimize(f::Function, g!::Function, h!::Function, initial_x::AbstractArray, options::Options)
    d = TwiceDifferentiable(f, g!, h!, initial_x)
    optimize(d, initial_x, fallback_method(d), options)
end
function optimize(f::Function, g!::Function, h!::Function, initial_x::AbstractArray, method::Optimizer, options::Options = Options())
    d = TwiceDifferentiable(f, g!, h!, initial_x)
    optimize(d, initial_x, method, options)
end

# OnceDifferentiable -> TwiceDifferentiable using autodiff
function optimize(d::OnceDifferentiable, initial_x::AbstractArray, method::SecondOrderSolver, options::Options = Options())
    optimize(TwiceDifferentiable(d), initial_x, method, options)
end
