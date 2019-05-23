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
x_tol::Real = 0.0,
f_tol::Real = 0.0,
g_tol::Real = 1e-8,
outer_x_tol::Real = 0.0,
outer_f_tol::Real = 0.0,
outer_g_tol::Real = 1e-8,
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
    x_tol::T
    f_tol::T
    g_tol::T
    outer_x_tol::T
    outer_f_tol::T
    outer_g_tol::T
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
        x_tol::Real = 0.0,
        f_tol::Real = 0.0,
        g_tol::Real = 1e-8,
        outer_x_tol::Real = 0.0,
        outer_f_tol::Real = 0.0,
        outer_g_tol::Real = 1e-8,
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
    Options(promote(x_tol, f_tol, g_tol, outer_x_tol, outer_f_tol, outer_g_tol)..., f_calls_limit, g_calls_limit, h_calls_limit,
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

mutable struct MultivariateOptimizationResults{O, T, Tx, Tc, Tf, M} <: OptimizationResults
    method::O
    initial_x::Tx
    minimizer::Tx
    minimum::Tf
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    x_tol::T
    x_abschange::Tc
    f_converged::Bool
    f_tol::T
    f_abschange::Tc
    g_converged::Bool
    g_tol::T
    g_residual::Tc
    f_increased::Bool
    trace::M
    f_calls::Int
    g_calls::Int
    h_calls::Int
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

    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r)
    if length(join(initial_state(r), ",")) < 40
        @printf io " * Starting Point: [%s]\n" join(initial_state(r), ",")
    else
        @printf io " * Starting Point: [%s, ...]\n" join(take(initial_state(r),
2), ",")
    end
    if length(join(minimizer(r), ",")) < 40
        @printf io " * Minimizer: [%s]\n" join(minimizer(r), ",")
    else
        @printf io " * Minimizer: [%s, ...]\n" join(take(minimizer(r), 2), ",")
    end
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: %s\n" converged(r)
    if isa(r.method, NelderMead)
        @printf io "   *  √(Σ(yᵢ-ȳ)²)/n < %.1e: %s\n" g_tol(r) g_converged(r)
    else
        @printf io "   * |x - x'| ≤ %.1e: %s \n" x_tol(r) x_converged(r)
        @printf io "     |x - x'| = %.2e \n"  x_abschange(r)
        @printf io "   * |f(x) - f(x')| ≤ %.1e |f(x)|: %s\n" f_tol(r) f_converged(r)
        @printf io "     |f(x) - f(x')| = %.2e |f(x)|\n" f_relchange(r)
        @printf io "   * |g(x)| ≤ %.1e: %s \n" g_tol(r) g_converged(r)
        @printf io "     |g(x)| = %.2e \n"  g_residual(r)
        @printf io "   * Stopped by an increasing objective: %s\n" (f_increased(r) && !iteration_limit_reached(r))
    end
    @printf io "   * Reached Maximum Number of Iterations: %s\n" iteration_limit_reached(r)
    @printf io " * Objective Calls: %d" f_calls(r)
    if !(isa(r.method, NelderMead) || isa(r.method, SimulatedAnnealing))
        @printf io "\n * Gradient Calls: %d" g_calls(r)
    end
    if isa(r.method, Newton) || isa(r.method, NewtonTrustRegion)
        @printf io "\n * Hessian Calls: %d" h_calls(r)
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
