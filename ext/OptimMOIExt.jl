module OptimMOIExt

using Optim
using Optim.LinearAlgebra: rmul! 
import MathOptInterface as MOI

function __init__()
    setglobal!(Optim, :Optimizer, Optimizer)
end

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    # Problem data.
    variables::MOI.Utilities.VariablesContainer{T}
    starting_values::Vector{Union{Nothing,T}}
    nlp_model::Union{MOI.Nonlinear.Model,Nothing}
    sense::MOI.OptimizationSense

    # Parameters.
    method::Union{Optim.AbstractOptimizer,Nothing}
    silent::Bool
    options::Dict{Symbol,Any}

    # Solution attributes.
    results::Union{Nothing,Optim.MultivariateOptimizationResults}
end

function Optimizer{T}() where {T}
    return Optimizer{T}(
        MOI.Utilities.VariablesContainer{T}(),
        Union{Nothing,T}[],
        nothing,
        MOI.FEASIBILITY_SENSE,
        nothing,
        false,
        Dict{Symbol,Any}(),
        nothing,
    )
end
Optimizer() = Optimizer{Float64}()

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.supports(::Optimizer, ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction})
    return true
end
MOI.supports(::Optimizer, ::MOI.Silent) = true
function MOI.supports(::Optimizer, p::MOI.RawOptimizerAttribute)
    return p.name == "method" || hasfield(Optim.Options, Symbol(p.name))
end

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

const BOUNDS{T} = Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T},MOI.Interval{T}}
const _SETS{T} = Union{MOI.GreaterThan{T},MOI.LessThan{T},MOI.EqualTo{T}}

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VariableIndex},
    ::Type{<:BOUNDS{T}},
) where {T}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.ScalarNonlinearFunction},
    ::Type{<:_SETS{T}},
) where {T}
    return true
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Optim"

function MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
    return
end
function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction{F}, func::F) where {F}
    nl = convert(MOI.ScalarNonlinearFunction, func)
    if isnothing(model.nlp_model)
        model.nlp_model = MOI.Nonlinear.Model()
    end
    MOI.Nonlinear.set_objective(model.nlp_model, nl)
    return nothing
end

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

const TIME_LIMIT = "time_limit"
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawOptimizerAttribute(TIME_LIMIT), Float64(value))
end
function MOI.set(model::Optimizer, attr::MOI.TimeLimitSec, ::Nothing)
    delete!(model.options, Symbol(TIME_LIMIT))
end
function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get(model.options, Symbol(TIME_LIMIT), nothing)
end

MOI.Utilities.map_indices(::Function, opt::Optim.AbstractOptimizer) = opt

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    if p.name == "method"
        model.method = value
    else
        model.options[Symbol(p.name)] = value
    end
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if p.name == "method"
        return p.method
    end
    key = Symbol(p.name)
    if haskey(model.options, key)
        return model.options[key]
    end
    error("RawOptimizerAttribute with name $(p.name) is not set.")
end

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = time_run(model.results)

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.variables)
    empty!(model.starting_values)
    model.nlp_model = nothing
    model.sense = MOI.FEASIBILITY_SENSE
    model.results = nothing
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.variables) &&
           isempty(model.starting_values) &&
           isnothing(model.nlp_model) &&
           model.sense == MOI.FEASIBILITY_SENSE
end

function MOI.add_variable(model::Optimizer{T}) where {T}
    push!(model.starting_values, nothing)
    return MOI.add_variable(model.variables)
end
function MOI.is_valid(model::Optimizer, index::Union{MOI.VariableIndex,MOI.ConstraintIndex})
    return MOI.is_valid(model.variables, index)
end

function MOI.add_constraint(
    model::Optimizer{T},
    vi::MOI.VariableIndex,
    set::BOUNDS{T},
) where {T}
    return MOI.add_constraint(model.variables, vi, set)
end

function MOI.add_constraint(
    model::Optimizer{T},
    f::MOI.ScalarNonlinearFunction,
    s::_SETS{T},
) where {T}
    if model.nlp_model === nothing
        model.nlp_model = MOI.Nonlinear.Model()
    end
    index = MOI.Nonlinear.add_constraint(model.nlp_model, f, s)
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(index.value)
end

function starting_value(optimizer::Optimizer{T}, i) where {T}
    start = optimizer.starting_values[i]
    v = optimizer.variables
    if isfinite(v.lower[i])
        if isfinite(v.upper[i])
            if !isnothing(start) && v.lower[i] < start < v.upper[i]
                return start
            else
                return (v.lower[i] + v.upper[i]) / 2
            end
        else
            if !isnothing(start) && v.lower[i] < start
                return start
            else
                return v.lower[i] + 1.0
            end
        end
    else
        if isfinite(v.upper[i])
            if !isnothing(start) && start < v.upper[i]
                return start
            else
                return v.upper[i] - 1.0
            end
        else
            return something(start, 0.0)
        end
    end
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, vi)
    model.starting_values[vi.value] = value
    return
end

function requested_features(::Optim.ZerothOrderOptimizer, has_constraints)
    return Symbol[]
end
function requested_features(::Optim.FirstOrderOptimizer, has_constraints)
    features = [:Grad]
    if has_constraints
        push!(features, :Jac)
    end
    return features
end
function requested_features(::Union{IPNewton,Optim.SecondOrderOptimizer}, has_constraints)
    features = [:Grad, :Hess]
    if has_constraints
        push!(features, :Jac)
    end
    return features
end

function sparse_to_dense!(A, I::Vector, nzval)
    for k in eachindex(I)
        i, j = I[k]
        A[i, j] += nzval[k]
    end
    return A
end

function sym_sparse_to_dense!(A, I::Vector, nzval)
    for k in eachindex(I)
        i, j = I[k]
        A[i, j] += nzval[k]
        A[j, i] = A[i, j]
    end
    return A
end

function MOI.optimize!(model::Optimizer{T}) where {T}
    backend = MOI.Nonlinear.SparseReverseMode()
    vars = MOI.get(model.variables, MOI.ListOfVariableIndices())
    evaluator = MOI.Nonlinear.Evaluator(model.nlp_model, backend, vars)
    nlp_data = MOI.NLPBlockData(evaluator)

    # load parameters
    if isnothing(model.nlp_model)
        error("An objective should be provided to Optim with `@objective`.")
    end
    objective_scale = model.sense == MOI.MAX_SENSE ? -one(T) : one(T)
    zero_μ = zeros(T, length(nlp_data.constraint_bounds))
    function f(x)
        return objective_scale * MOI.eval_objective(evaluator, x)
    end
    function g!(G, x)
        fill!(G, zero(T))
        MOI.eval_objective_gradient(evaluator, G, x)
        if model.sense == MOI.MAX_SENSE
            rmul!(G, objective_scale)
        end
        return G
    end
    function h!(H, x)
        fill!(H, zero(T))
        MOI.eval_hessian_lagrangian(evaluator, H_nzval, x, objective_scale, zero_μ)
        sym_sparse_to_dense!(H, hessian_structure, H_nzval)
        return H
    end

    method = model.method
    nl_constrained = !isempty(nlp_data.constraint_bounds)
    features = MOI.features_available(evaluator)
    has_bounds = any(
        vi ->
            isfinite(model.variables.lower[vi.value]) ||
                isfinite(model.variables.upper[vi.value]),
        vars,
    )
    if method === nothing
        if nl_constrained
            method = IPNewton()
        elseif :Grad in features
            # FIXME `fallback_method(f, g!, h!)` returns `Newton` but if there
            # are variable bounds, `Newton` is not supported. On the other hand,
            # `fallback_method(f, g!)` returns `LBFGS` which is supported if `has_bounds`.
            if :Hess in features && !has_bounds
                method = Optim.fallback_method(f, g!, h!)
            else
                method = Optim.fallback_method(f, g!)
            end
        else
            method = Optim.fallback_method(f)
        end
    end
    used_features = requested_features(method, nl_constrained)
    MOI.initialize(evaluator, used_features)

    if :Hess in used_features
        hessian_structure = MOI.hessian_lagrangian_structure(evaluator)
        H_nzval = zeros(T, length(hessian_structure))
    end

    initial_x = starting_value.(model, eachindex(model.starting_values))
    options = copy(model.options)
    if !nl_constrained && has_bounds && !(method isa IPNewton)
        options = Optim.Options(; options...)
        model.results = optimize(
            f,
            g!,
            model.variables.lower,
            model.variables.upper,
            initial_x,
            Fminbox(method),
            options;
            inplace = true,
        )
    else
        d = Optim.promote_objtype(method, initial_x, Optim.DEFAULT_AD_TYPE, true, f, g!, h!)
        options = Optim.Options(; Optim.default_options(method)..., options...)
        if nl_constrained || has_bounds
            if nl_constrained
                lc = [b.lower for b in nlp_data.constraint_bounds]
                uc = [b.upper for b in nlp_data.constraint_bounds]
                c!(c, x) = MOI.eval_constraint(evaluator, c, x)
                if !(:Jac in features)
                    error(
                        "Nonlinear constraints should be differentiable to be used with Optim.",
                    )
                end
                if !(:Hess in features)
                    error(
                        "Nonlinear constraints should be twice differentiable to be used with Optim.",
                    )
                end
                jacobian_structure = MOI.jacobian_structure(evaluator)
                J_nzval = zeros(T, length(jacobian_structure))
                function jacobian!(J, x)
                    fill!(J, zero(T))
                    MOI.eval_constraint_jacobian(evaluator, J_nzval, x)
                    sparse_to_dense!(J, jacobian_structure, J_nzval)
                    return J
                end
                function con_hessian!(H, x, λ)
                    fill!(H, zero(T))
                    MOI.eval_hessian_lagrangian(evaluator, H_nzval, x, zero(T), λ)
                    sym_sparse_to_dense!(H, hessian_structure, H_nzval)
                    return H
                end
                c = TwiceDifferentiableConstraints(
                    c!,
                    jacobian!,
                    con_hessian!,
                    model.variables.lower,
                    model.variables.upper,
                    lc,
                    uc,
                )
            else
                @assert has_bounds
                c = TwiceDifferentiableConstraints(
                    model.variables.lower,
                    model.variables.upper,
                )
            end
            model.results = optimize(d, c, initial_x, method, options)
        else
            model.results = optimize(d, initial_x, method, options)
        end
    end
    return
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.results === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    elseif Optim.converged(model.results)
        return MOI.LOCALLY_SOLVED
    else
        return MOI.OTHER_ERROR
    end
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    return summary(model.results)
end

# Ipopt always has an iterate available.
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.results === nothing ? 0 : 1
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    if Optim.converged(model.results)
        return MOI.FEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end
MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    val = Optim.minimum(model.results)
    if model.sense == MOI.MAX_SENSE
        val = -val
    end
    return val
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return Optim.minimizer(model.results)[vi.value]
end

function MOI.get(
    model::Optimizer{T},
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:BOUNDS{T}},
) where {T}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return Optim.minimizer(model.results)[ci.value]
end
end # module
