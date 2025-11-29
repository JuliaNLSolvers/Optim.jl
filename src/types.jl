abstract type AbstractOptimizer end
abstract type AbstractConstrainedOptimizer <: AbstractOptimizer end
abstract type ZerothOrderOptimizer <: AbstractOptimizer end
abstract type FirstOrderOptimizer <: AbstractOptimizer end
abstract type SecondOrderOptimizer <: AbstractOptimizer end
abstract type UnivariateOptimizer <: AbstractOptimizer end

abstract type AbstractOptimizerState end
abstract type ZerothOrderState <: AbstractOptimizerState end

"""
    Options(; opts...)

Specify configurable optimizer options `opts...`. Unspecified options are set to the default
values below (values 0 and NaN indicate unlimited):

```
x_abstol::Real = 0.0,
x_reltol::Real = 0.0,
f_abstol::Real = 0.0,
f_reltol::Real = 0.0,
g_abstol::Real = 1e-8,
outer_x_abstol::Real = 0.0,
outer_x_reltol::Real = 0.0,
outer_f_abstol::Real = 0.0,
outer_f_reltol::Real = 0.0,
outer_g_abstol::Real = 1e-8,
f_calls_limit::Int = 0,
g_calls_limit::Int = 0,
h_calls_limit::Int = 0,
allow_f_increases::Bool = true,
allow_outer_f_increases::Bool = true,
successive_f_tol::Int = 1,
iterations::Int = 1_000,
outer_iterations::Int = 1000,
store_trace::Bool = false,
show_trace::Bool = false,
extended_trace::Bool = false,
show_warnings::Bool = true,
show_every::Int = 1,
callback = nothing,
time_limit = NaN
```

It is also possible to pass a previously defined `Options` argument as the first argument,
i.e., as:

```jl
    Options(inherit_options; opts...)
```

Default values for unspecified `opts` will then be "inherited" from `inherit_options`. This
can be used to modify a subset of options in a previously defined `Options` variable.

For more information on individual options, see the documentaton at
http://julianlsolvers.github.io/Optim.jl/stable/#user/config/.
"""
struct Options{T, TCallback}
    x_abstol::T
    x_reltol::T
    f_abstol::T
    f_reltol::T
    g_abstol::T
    outer_x_abstol::T
    outer_x_reltol::T
    outer_f_abstol::T
    outer_f_reltol::T
    outer_g_abstol::T
    f_calls_limit::Int
    g_calls_limit::Int
    h_calls_limit::Int
    allow_f_increases::Bool
    allow_outer_f_increases::Bool
    successive_f_tol::Int
    iterations::Int
    outer_iterations::Int
    store_trace::Bool
    trace_simplex::Bool
    show_trace::Bool
    extended_trace::Bool
    show_warnings::Bool
    show_every::Int
    callback::TCallback
    time_limit::Float64
end

function Options(;
    x_tol = nothing,
    f_tol = nothing,
    g_tol = nothing,
    x_abstol::Real = 0.0,
    x_reltol::Real = 0.0,
    f_abstol::Real = 0.0,
    f_reltol::Real = 0.0,
    g_abstol::Real = 1e-8,
    outer_x_tol = nothing,
    outer_f_tol = nothing,
    outer_g_tol = nothing,
    outer_x_abstol::Real = 0.0,
    outer_x_reltol::Real = 0.0,
    outer_f_abstol::Real = 0.0,
    outer_f_reltol::Real = 0.0,
    outer_g_abstol::Real = 1e-8,
    f_calls_limit::Int = 0,
    g_calls_limit::Int = 0,
    h_calls_limit::Int = 0,
    allow_f_increases::Bool = true,
    allow_outer_f_increases::Bool = true,
    successive_f_tol::Int = 1,
    iterations::Int = 1_000,
    outer_iterations::Int = 1000,
    store_trace::Bool = false,
    trace_simplex::Bool = false,
    show_trace::Bool = false,
    extended_trace::Bool = false,
    show_warnings::Bool = true,
    show_every::Int = 1,
    callback = nothing,
    time_limit = NaN,
)
    show_every = show_every > 0 ? show_every : 1
    #if extended_trace && callback === nothing
    #    show_trace = true
    #end
    if !(x_tol === nothing)
        @warn(
            lazy"x_tol is deprecated. Use x_abstol or x_reltol instead. The provided value ($(x_tol)) will be used as x_abstol.",
        )
        x_abstol = x_tol
    end
    if !(g_tol === nothing)
        # lets deprecate this when reltol is introduced
        g_abstol = g_tol
    end
    if !(f_tol === nothing)
        @warn(
            lazy"f_tol is deprecated. Use f_abstol or f_reltol instead. The provided value ($(f_tol)) will be used as f_reltol.",
        )
        f_reltol = f_tol
    end
    if !(outer_x_tol === nothing)
        @warn(
            lazy"outer_x_tol is deprecated. Use outer_x_abstol or outer_x_reltol instead. The provided value ($(outer_x_tol)) will be used as x_abstol.",
        )
        outer_x_abstol = outer_x_tol
    end
    if !(outer_g_tol === nothing)
       @warn(
            lazy"outer_g_tol is deprecated. Use outer_g_abstol instead. The provided value ($(outer_g_abstol)) will be used as x_abstol.",
        )
         outer_g_abstol = outer_g_tol
    end
    if !(outer_f_tol === nothing)
        @warn(
            lazy"outer_f_tol is deprecated. Use outer_f_abstol or outer_f_reltol instead. The provided value ($(outer_f_tol)) will be used as outer_f_reltol.",
        )
         outer_f_reltol = outer_f_tol
    end
    Options(
        promote(
            x_abstol,
            x_reltol,
            f_abstol,
            f_reltol,
            g_abstol,
            outer_x_abstol,
            outer_x_reltol,
            outer_f_abstol,
            outer_f_reltol,
            outer_g_abstol,
        )...,
        f_calls_limit,
        g_calls_limit,
        h_calls_limit,
        allow_f_increases,
        allow_outer_f_increases,
        successive_f_tol,
        Int(iterations),
        Int(outer_iterations),
        store_trace,
        trace_simplex,
        show_trace,
        extended_trace,
        show_warnings,
        Int(show_every),
        callback,
        Float64(time_limit),
    )
end

function Options(o::Options; kws...)
    o_nt = (; (name => getproperty(o, name) for name in propertynames(o))...)
    return Options(; o_nt..., kws...)
end

_show_helper(output, k, v) = output * "$k = $v, "
_show_helper(output, k, ::Nothing) = output

function Base.show(io::IO, o::Options)
    content = foldl(fieldnames(typeof(o)), init = "Optim.Options(") do output, k
        v = getfield(o, k)
        return _show_helper(output, k, v)
    end
    print(io, content)
    println(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", o::Options)
    for k in fieldnames(typeof(o))
        v = getfield(o, k)
        if v isa Nothing
            @printf io "%24s = %s\n" k "nothing"
        else
            @printf io "%24s = %s\n" k v
        end
    end
end

function print_header(options::Options)
    if options.show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
end

function print_header(method::AbstractOptimizer)
    @printf "Iter     Function value   Gradient norm \n"
end

struct OptimizationState{Tf<:Real,T<:AbstractOptimizer}
    iteration::Int
    value::Tf
    g_norm::Tf
    metadata::Dict
end

const OptimizationTrace{Tf,T} = Vector{OptimizationState{Tf,T}}

"Termination codes for Optim.jl."
@enumx TerminationCode begin
    "Nelder-Mead simplex converged."
    NelderMeadCriterion
    "First (partial) derivative had a magnitude below the prescribed tolerance."
    GradientNorm
    "The change in optimization variables was zero (the tolerance was not set by the user)."
    NoXChange
    "The change in the objective was zero (the tolerance was not set by the user)."
    NoObjectiveChange
    "The change in the optimization variables was below the prescribed tolerance."
    SmallXChange
    "The change in the objective was below the prescribed tolerance."
    SmallObjectiveChange
    "The line search failed to find a point that decreased the objective."
    FailedLinesearch
    "User callback returned `true`."
    Callback
    "The number of iterations exceeded the maximum number allowed."
    Iterations
    "Time budget was exceeded."
    Time
    "Objective function evaluations exceeded the maximum number allowed."
    ObjectiveCalls
    "Gradient evaluations exceeded the maximum number allowed."
    GradientCalls
    "Hessian evaluations exceeded the maximum number allowed."
    HessianCalls
    "Objective function value increased."
    ObjectiveIncreased
    "Objective function was not finite"
    ObjectiveNotFinite
    "Gradient was not finite"
    GradientNotFinite
    "Hessian was not finite"
    HessianNotFinite
    "For algorithms where the TerminationCode is not yet implemented."
    NotImplemented
end

abstract type OptimizationResults end

mutable struct MultivariateOptimizationResults{O,Tx,Tc,Tf,M,Tsb} <: OptimizationResults
    method::O
    initial_x::Tx
    minimizer::Tx
    minimum::Tf
    iterations::Int
    x_abstol::Tf
    x_reltol::Tf
    x_abschange::Tc
    x_relchange::Tc
    f_abstol::Tf
    f_reltol::Tf
    f_abschange::Tc
    f_relchange::Tc
    g_abstol::Tf
    g_residual::Tc
    trace::M
    f_calls::Int
    g_calls::Int
    h_calls::Int
    time_limit::Float64
    time_run::Float64
    stopped_by::Tsb
    termination_code::TerminationCode.T
end


termination_code(mvr::MultivariateOptimizationResults) = mvr.termination_code

# pick_best_x and pick_best_f are used to pick the minimizer if we stopped because
# f increased and we didn't allow it
pick_best_x(f_increased::Bool, state::AbstractOptimizerState) = f_increased ? state.x_previous : state.x
pick_best_f(f_increased::Bool, state::AbstractOptimizerState) = f_increased ? state.f_x_previous : state.f_x

function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.value t.g_norm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

function Base.show(io::IO, tr::OptimizationTrace)
    @printf io "Iter     Function value   Gradient norm \n"
    @printf io "------   --------------   --------------\n"
    for state in tr
        show(io, state)
    end
    return
end

function Base.show(io::IO, r::MultivariateOptimizationResults)
    print(io, " * Status: ")
    if converged(r)
        print(io, "success")
    else
        print(io, "failure")
    end
    if iteration_limit_reached(r)
        print(io, " (reached maximum number of iterations)")
    elseif f_increased(r)
        print(io, " (objective increased between iterations)")
    end
    if isa(r.stopped_by.ls_failed, Bool) && r.stopped_by.ls_failed
        print(io, " (line search failed)")
    end
    if time_run(r) > time_limit(r)
        print(io, " (exceeded time limit of ", time_limit(r), ")")
    end

    println(io, "\n\n * Candidate solution")
    @printf io "    Final objective value:     %e" minimum(r)

    println(io, "\n\n * Found with")
    print(io, "    Algorithm:     ")
    summary(io, r)

    println(io, "\n\n * Convergence measures")
    if isa(r.method, NelderMead)
        @printf io "    √(Σ(yᵢ-ȳ)²)/n %s %.1e\n" g_converged(r) ? "≤" : "≰" g_tol(r)
    else
        @printf io "    |x - x'|               = %.2e %s %.1e\n" x_abschange(r) x_abschange(
            r,
        ) <= x_abstol(
            r,
        ) ? "≤" : "≰" x_abstol(r)
        @printf io "    |x - x'|/|x'|          = %.2e %s %.1e\n" x_relchange(r) x_relchange(
            r,
        ) <= x_reltol(
            r,
        ) ? "≤" : "≰" x_reltol(r)
        @printf io "    |f(x) - f(x')|         = %.2e %s %.1e\n" f_abschange(r) f_abschange(
            r,
        ) <= f_abstol(
            r,
        ) ? "≤" : "≰" f_abstol(r)
        @printf io "    |f(x) - f(x')|/|f(x')| = %.2e %s %.1e\n" f_relchange(r) f_relchange(
            r,
        ) <= f_reltol(
            r,
        ) ? "≤" : "≰" f_reltol(r)
        @printf io "    |g(x)|                 = %.2e %s %.1e\n" g_residual(r) g_residual(
            r,
        ) <= g_tol(r) ? "≤" : "≰" g_tol(r)
    end

    @printf io "\n"

    @printf io " * Work counters\n"
    @printf io "    Seconds run:   %d  (vs limit %d)\n" time_run(r) isnan(time_limit(r)) ?
                                                                    Inf : time_limit(r)
    @printf io "    Iterations:    %d\n" iterations(r)
    @printf io "    f(x) calls:    %d\n" f_calls(r)
    if !(isa(r.method, NelderMead) || isa(r.method, SimulatedAnnealing))
        @printf io "    ∇f(x) calls:   %d\n" g_calls(r)
    end
    if isa(r.method, Newton) || isa(r.method, NewtonTrustRegion)
        @printf io "    ∇²f(x) calls:  %d\n" h_calls(r)
    end
    return
end


function Base.append!(
    a::MultivariateOptimizationResults,
    b::MultivariateOptimizationResults,
)
    a.iterations += iterations(b)
    a.minimizer = minimizer(b)
    a.minimum = minimum(b)
    a.stopped_by = b.stopped_by
    append!(a.trace, b.trace)
    a.f_calls += f_calls(b)
    a.g_calls += g_calls(b)
end
