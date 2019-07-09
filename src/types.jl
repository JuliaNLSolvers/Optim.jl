abstract type AbstractOptimizer end
abstract type AbstractConstrainedOptimizer end
abstract type ZerothOrderOptimizer <: AbstractOptimizer end
abstract type FirstOrderOptimizer  <: AbstractOptimizer end
abstract type SecondOrderOptimizer <: AbstractOptimizer end
abstract type UnivariateOptimizer  <: AbstractOptimizer end

abstract type AbstractOptimizerState end
abstract type ZerothOrderState <: AbstractOptimizerState end

"""
Configurable options with defaults (values 0 and NaN indicate unlimited):
```
x_abstol::Real = 0.0,
x_reltol::Real = 0.0,
f_abstol::Real = 0.0,
f_reltol::Real = 0.0,
g_abstol::Real = 1e-8,
g_reltol::Real = 1e-8,
outer_x_abstol::Real = 0.0,
outer_x_reltol::Real = 0.0,
outer_f_abstol::Real = 0.0,
outer_f_reltol::Real = 0.0,
outer_g_abstol::Real = 1e-8,
outer_g_reltol::Real = 1e-8,
f_calls_limit::Int = 0,
g_calls_limit::Int = 0,
h_calls_limit::Int = 0,
allow_f_increases::Bool = false,
allow_outer_f_increases::Bool = false,
successive_f_tol::Int = 0,
iterations::Int = 1_000,
outer_iterations::Int = 1000,
store_trace::Bool = false,
show_trace::Bool = false,
extended_trace::Bool = false,
show_every::Int = 1,
callback = nothing,
time_limit = NaN
```
See http://julianlsolvers.github.io/Optim.jl/stable/#user/config/
"""
struct Options{T, TCallback}
    x_abstol::T
    x_reltol::T
    f_abstol::T
    f_reltol::T
    g_abstol::T
    g_reltol::T
    outer_x_abstol::T
    outer_x_reltol::T
    outer_f_abstol::T
    outer_f_reltol::T
    outer_g_abstol::T
    outer_g_reltol::T
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
        g_reltol::Real = 1e-8,
        outer_x_tol = 0.0,
        outer_f_tol = 0.0,
        outer_g_tol = 1e-8,
        outer_x_abstol::Real = 0.0,
        outer_x_reltol::Real = 0.0,
        outer_f_abstol::Real = 0.0,
        outer_f_reltol::Real = 0.0,
        outer_g_abstol::Real = 1e-8,
        outer_g_reltol::Real = 1e-8,
        f_calls_limit::Int = 0,
        g_calls_limit::Int = 0,
        h_calls_limit::Int = 0,
        allow_f_increases::Bool = false,
        allow_outer_f_increases::Bool = false,
        successive_f_tol::Int = 0,
        iterations::Int = 1_000,
        outer_iterations::Int = 1000,
        store_trace::Bool = false,
        trace_simplex::Bool = false,
        show_trace::Bool = false,
        extended_trace::Bool = false,
        show_every::Int = 1,
        callback = nothing,
        time_limit = NaN)
    show_every = show_every > 0 ? show_every : 1
    #if extended_trace && callback == nothing
    #    show_trace = true
    #end
    if !(x_tol == nothing)
        x_abstol = x_tol
    end
    if !(g_tol == nothing)
        g_abstol = g_tol
    end
    if !(f_tol == nothing)
        f_reltol = f_tol
    end
    if !(outer_x_tol == nothing)
        outer_x_abstol = outer_x_tol
    end
    if !(outer_g_tol == nothing)
        outer_g_abstol = outer_g_tol
    end
    if !(outer_f_tol == nothing)
        outer_f_reltol = outer_f_tol
    end
    Options(promote(x_abstol, x_reltol, f_abstol, f_reltol, g_abstol, g_reltol, outer_x_abstol, outer_x_reltol, outer_f_abstol, outer_f_reltol, outer_g_abstol, outer_g_reltol)..., f_calls_limit, g_calls_limit, h_calls_limit,
        allow_f_increases, allow_outer_f_increases, successive_f_tol, Int(iterations), Int(outer_iterations), store_trace, trace_simplex, show_trace, extended_trace,
        Int(show_every), callback, Float64(time_limit))
end

function Base.show(io::IO, o::Optim.Options)
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

struct OptimizationState{Tf, T <: AbstractOptimizer}
    iteration::Int
    value::Tf
    g_norm::Tf
    metadata::Dict
end

const OptimizationTrace{Tf, T} = Vector{OptimizationState{Tf, T}}

abstract type OptimizationResults end

mutable struct MultivariateOptimizationResults{O, T, Tx, Tc, Tf, M, Tls} <: OptimizationResults
    method::O
    initial_x::Tx
    minimizer::Tx
    minimum::Tf
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    x_abstol::T
    x_reltol::T
    x_abschange::Tc
    x_relchange::Tc
    f_converged::Bool
    f_abstol::T
    f_reltol::T
    f_abschange::Tc
    f_relchange::Tc
    g_converged::Bool
    g_abstol::T
    g_residual::Tc
    f_increased::Bool
    trace::M
    f_calls::Int
    g_calls::Int
    h_calls::Int
    ls_success::Tls
end
# pick_best_x and pick_best_f are used to pick the minimizer if we stopped because
# f increased and we didn't allow it
pick_best_x(f_increased, state) = f_increased ? state.x_previous : state.x
pick_best_f(f_increased, state, d) = f_increased ? state.f_x_previous : value(d)

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
    take = Iterators.take
    failure_string = "failure"
    if iteration_limit_reached(r)
        failure_string *= " (reached maximum number of iterations)"
    end
    if f_increased(r) && !iteration_limit_reached(r)
        failure_string *= " (objective increased between iterations)"
    end
    if isa(r.ls_success, Bool) && !r.ls_success
        failure_string *= " (line search failed)"
    end
    @printf io " * Status: %s\n\n" converged(r) ? "success" : failure_string

    @printf io " * Candidate solution\n"
    nx = length(minimizer(r))
    str_x_elements = [@sprintf "%.2e" _x for _x in take(minimizer(r), min(nx, 3))]
    if nx >= 4
        push!(str_x_elements, " ...")
    end

    @printf io "    Minimizer: [%s]\n" join(str_x_elements, ", ")

    @printf io "    Minimum:   %e\n" minimum(r)
    @printf io "\n"

    @printf io " * Found with\n"
    @printf io "    Algorithm:     %s\n" summary(r)
    nx = length(initial_state(r))
    str_x_elements = [@sprintf "%.2e" _x for _x in take(initial_state(r), min(nx, 3))]
    if nx >= 4
        push!(str_x_elements, " ...")
    end

    @printf io "    Initial Point: [%s]\n" join(str_x_elements, ", ")

    @printf io "\n"
    @printf io " * Convergence measures\n"
    if isa(r.method, NelderMead)
        @printf io "    √(Σ(yᵢ-ȳ)²)/n %s %.1e\n" g_converged(r) ? "≤" : "≰" g_tol(r)
    else
        @printf io "    |x - x'|               = %.2e %s %.1e\n"  x_abschange(r) x_abschange(r)<=x_abstol(r) ? "≤" : "≰" x_abstol(r)
        @printf io "    |x - x'|/|x'|          = %.2e %s %.1e\n"  x_relchange(r) x_relchange(r)<=x_reltol(r) ? "≤" : "≰" x_reltol(r)
        @printf io "    |f(x) - f(x')|         = %.2e %s %.1e\n"  f_abschange(r) f_abschange(r)<=f_abstol(r) ? "≤" : "≰" f_abstol(r)
        @printf io "    |f(x) - f(x')|/|f(x')| = %.2e %s %.1e\n"  f_relchange(r) f_relchange(r)<=f_reltol(r) ? "≤" : "≰" f_reltol(r)
        @printf io "    |g(x)|                 = %.2e %s %.1e\n" g_residual(r) g_residual(r)<=g_tol(r) ?  "≤" : "≰" g_tol(r)
    end

    @printf io "\n"

    @printf io " * Work counters\n"
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


function Base.append!(a::MultivariateOptimizationResults, b::MultivariateOptimizationResults)
    a.iterations += iterations(b)
    a.minimizer = minimizer(b)
    a.minimum = minimum(b)
    a.iteration_converged = iteration_limit_reached(b)
    a.x_converged = x_converged(b)
    a.f_converged = f_converged(b)
    a.g_converged = g_converged(b)
    append!(a.trace, b.trace)
    a.f_calls += f_calls(b)
    a.g_calls += g_calls(b)
end
