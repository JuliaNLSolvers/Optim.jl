import MathOptInterface
const MOI = MathOptInterface

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    # Problem data.
    variables::MOI.Utilities.VariablesContainer{T}
    starting_values::Vector{Union{Nothing,T}}
    nlp_data::Union{MOI.NLPBlockData,Nothing}
    sense::MOI.OptimizationSense

    # Parameters.
    method::Union{AbstractOptimizer,Nothing}
    silent::Bool
    options::Dict{Symbol,Any}

    # Solution attributes.
    results::Union{Nothing,MultivariateOptimizationResults}
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

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.Silent) = true
function MOI.supports(::Optimizer, p::MOI.RawOptimizerAttribute)
    return p.name == "method" || hasfield(Options, Symbol(p.name))
end

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

const BOUNDS{T} = Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T},MOI.Interval{T}}

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VariableIndex},
    ::Type{<:BOUNDS{T}},
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

MOI.Utilities.map_indices(::Function, opt::AbstractOptimizer) = opt

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
    model.nlp_data = nothing
    model.sense = MOI.FEASIBILITY_SENSE
    model.results = nothing
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.variables) &&
           isempty(model.starting_values) &&
           model.nlp_data === nothing &&
           model.sense == MOI.FEASIBILITY_SENSE
end

function MOI.add_variable(model::Optimizer{T}) where {T}
    push!(model.starting_values, NaN)
    return MOI.add_variable(model.variables)
end
function MOI.is_valid(model::Optimizer, index::Union{MOI.VariableIndex,MOI.ConstraintIndex})
    return MOI.is_valid(model.variables, index)
end

function MOI.add_constraint(model::Optimizer{T}, vi::MOI.VariableIndex, set::BOUNDS{T}) where {T}
    return MOI.add_constraint(model.variables, vi, set)
end

function starting_value(optimizer::Optimizer{T}, i) where {T}
    if optimizer.starting_values[i] !== nothing
        return optimizer.starting_values[i]
    else
        v = optimizer.variables
        return min(max(zero(T), v.lower[i]), v.upper[i])
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

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    return
end

function requested_features(::ZerothOrderOptimizer, has_constraints)
    return Symbol[]
end
function requested_features(::FirstOrderOptimizer, has_constraints)
    features = [:Grad]
    if has_constraints
        push!(features, :Jac)
    end
    return features
end
function requested_features(::Union{IPNewton,SecondOrderOptimizer}, has_constraints)
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
    num_variables = length(model.variable_info)

    # load parameters
    if model.nlp_data === nothing || !model.nlp_data.has_objective
        error("An objective should be provided to Optim with `@NLobjective`.")
    end
    objective_scale = model.sense == MOI.MAX_SENSE ? -one(T) : one(T)
    zero_μ = zeros(T, length(model.nlp_data.constraint_bounds))
    function f(x)
        return objective_scale * MOI.eval_objective(model.nlp_data.evaluator, x)
    end
    function g!(G, x)
        fill!(G, zero(T))
        MOI.eval_objective_gradient(model.nlp_data.evaluator, G, x)
        if model.sense == MOI.MAX_SENSE
            rmul!(G, objective_scale)
        end
        return G
    end
    function h!(H, x)
        fill!(H, zero(T))
        MOI.eval_hessian_lagrangian(model.nlp_data.evaluator, H_nzval, x, objective_scale, zero_μ)
        sym_sparse_to_dense!(H, hessian_structure, H_nzval)
        return H
    end

    method = model.method
    nl_constrained = !isempty(model.nlp_data.constraint_bounds)
    features = MOI.features_available(model.nlp_data.evaluator)
    if method === nothing
        if nl_constrained
            method = IPNewton()
        elseif :Grad in features
            if :Hess in features
                method = fallback_method(f, g!, h!)
            else
                method = fallback_method(f, g!)
            end
        else
            method = fallback_method(f)
        end
    end
    used_features = requested_features(method, nl_constrained)
    MOI.initialize(model.nlp_data.evaluator, used_features)

    if :Hess in used_features
        hessian_structure = MOI.hessian_lagrangian_structure(model.nlp_data.evaluator)
        H_nzval = zeros(T, length(hessian_structure))
    end

    initial_x = starting_value.(model, eachindex(model.starting_values))
    options = copy(model.options)
    has_bounds = any(info -> info.is_fixed || info.has_upper_bound || info.has_lower_bound, model.variable_info)
    if has_bounds
        lower = [info.lower_bound for info in model.variable_info]
        upper = [info.upper_bound for info in model.variable_info]
    end
    if !nl_constrained && has_bounds && !(method isa IPNewton)
        options = Options(; options...)
        model.results = optimize(f, g!, lower, upper, initial_x, Fminbox(method), options; inplace = true)
    else
        d = promote_objtype(method, initial_x, :finite, true, f, g!, h!)
        add_default_opts!(options, method)
        options = Options(; options...)
        if nl_constrained || has_bounds
            if nl_constrained
                lc = [b.lower for b in model.nlp_data.constraint_bounds]
                uc = [b.upper for b in model.nlp_data.constraint_bounds]
                c!(c, x) = MOI.eval_constraint(model.nlp_data.evaluator, c, x)
                if !(:Jac in features)
                    error("Nonlinear constraints should be differentiable to be used with Optim.")
                end
                if !(:Hess in features)
                    error("Nonlinear constraints should be twice differentiable to be used with Optim.")
                end
                jacobian_structure = MOI.jacobian_structure(model.nlp_data.evaluator)
                J_nzval = zeros(T, length(jacobian_structure))
                function jacobian!(J, x)
                    fill!(J, zero(T))
                    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, J_nzval, x)
                    sparse_to_dense!(J, jacobian_structure, J_nzval)
                    return J
                end
                function con_hessian!(H, x, λ)
                    fill!(H, zero(T))
                    MOI.eval_hessian_lagrangian(model.nlp_data.evaluator, H_nzval, x, zero(T), λ)
                    sym_sparse_to_dense!(H, hessian_structure, H_nzval)
                    return H
                end
                if !has_bounds
                    lower = fill(typemin(T), num_variables)
                    upper = fill(typemax(T), num_variables)
                end
                c = TwiceDifferentiableConstraints(
                    c!, jacobian!, con_hessian!,
                    lower, upper, lc, uc,
                )
            else
                @assert has_bounds
                c = TwiceDifferentiableConstraints(lower, upper)
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
    elseif converged(model.results)
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
    if converged(model.results)
        return MOI.FEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end
MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    val = minimum(model.results)
    if model.sense == MOI.MAX_SENSE
        val = -val
    end
    return val
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return minimizer(model.results)[vi.value]
end

function MOI.get(
    model::Optimizer{T},
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:BOUNDS{T}},
) where {T}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return minimizer(model.results)[ci.value]
end
