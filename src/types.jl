immutable OptimizationState
    iteration::Int
    value::Float64
    gradnorm::Float64
    metadata::Dict
end

function OptimizationState(i::Integer, f::Real)
    OptimizationState(int(i), @compat(Float64(f)), NaN, Dict())
end

function OptimizationState(i::Integer, f::Real, g::Real)
    OptimizationState(int(i), @compat(Float64(f)), @compat(Float64(g)), Dict())
end

immutable OptimizationTrace
    states::Vector{OptimizationState}
end

OptimizationTrace() = OptimizationTrace(Array(OptimizationState, 0))

abstract OptimizationResults

type MultivariateOptimizationResults{T,N} <: OptimizationResults
    method::ASCIIString
    initial_x::Array{T,N}
    minimum::Array{T,N}
    f_minimum::Float64
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    xtol::Float64
    f_converged::Bool
    ftol::Float64
    gr_converged::Bool
    grtol::Float64
    trace::OptimizationTrace
    f_calls::Int
    g_calls::Int
end

type UnivariateOptimizationResults{T} <: OptimizationResults
    method::ASCIIString
    initial_lower::T
    initial_upper::T
    minimum::T
    f_minimum::Float64
    iterations::Int
    converged::Bool
    rel_tol::Float64
    abs_tol::Float64
    trace::OptimizationTrace
    f_calls::Int
end

immutable DifferentiableFunction
    f::Function
    g!::Function
    fg!::Function
end

immutable TwiceDifferentiableFunction
    f::Function
    g!::Function
    fg!::Function
    h!::Function
end

function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.value t.gradnorm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

Base.push!(t::OptimizationTrace, s::OptimizationState) = push!(t.states, s)

Base.getindex(t::OptimizationTrace, i::Integer) = getindex(t.states, i)

function Base.setindex!(t::OptimizationTrace,
                        s::OptimizationState,
                        i::Integer)
    setindex!(t.states, s, i)
end

function Base.show(io::IO, t::OptimizationTrace)
    @printf io "Iter     Function value   Gradient norm \n"
    @printf io "------   --------------   --------------\n"
    for state in t.states
        show(io, state)
    end
    return
end

function converged(r::MultivariateOptimizationResults)
    return r.x_converged || r.f_converged || r.gr_converged
end

function Base.show(io::IO, r::MultivariateOptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" r.method
    @printf io " * Starting Point: [%s]\n" join(r.initial_x, ",")
    @printf io " * Minimum: [%s]\n" join(r.minimum, ",")
    @printf io " * Value of Function at Minimum: %f\n" r.f_minimum
    @printf io " * Iterations: %d\n" r.iterations
    @printf io " * Convergence: %s\n" converged(r)
    @printf io "   * |x - x'| < %.1e: %s\n" r.xtol r.x_converged
    @printf io "   * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" r.ftol r.f_converged
    @printf io "   * |g(x)| < %.1e: %s\n" r.grtol r.gr_converged
    @printf io "   * Exceeded Maximum Number of Iterations: %s\n" r.iteration_converged
    @printf io " * Objective Function Calls: %d\n" r.f_calls
    @printf io " * Gradient Call: %d" r.g_calls
    return
end

function converged(r::UnivariateOptimizationResults)
    return r.converged
end

function Base.show(io::IO, r::UnivariateOptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" r.method
    @printf io " * Search Interval: [%f, %f]\n" r.initial_lower r.initial_upper
    @printf io " * Minimum: %f\n" r.minimum
    @printf io " * Value of Function at Minimum: %f\n" r.f_minimum
    @printf io " * Iterations: %d\n" r.iterations
    @printf io " * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(%.1e*|x|+%.1e): %s\n" r.rel_tol r.abs_tol r.converged
    @printf io " * Objective Function Calls: %d" r.f_calls
    return
end

function Base.append!(a::MultivariateOptimizationResults, b::MultivariateOptimizationResults)
    a.iterations += b.iterations
    a.minimum = b.minimum
    a.f_minimum = b.f_minimum
    a.iteration_converged = b.iteration_converged
    a.x_converged = b.x_converged
    a.f_converged = b.f_converged
    a.gr_converged = b.gr_converged
    append!(a.trace, b.trace)
    a.f_calls += b.f_calls
    a.g_calls += b.g_calls
end

# TODO: Expose ability to do forward and backward differencing
function DifferentiableFunction(f::Function)
    function g!(x::Array, storage::Array)
        Calculus.finite_difference!(f, x, storage, :central)
        return
    end
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    return DifferentiableFunction(f, g!, fg!)
end

function DifferentiableFunction(f::Function, g!::Function)
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    return DifferentiableFunction(f, g!, fg!)
end

function TwiceDifferentiableFunction(f::Function)
    function g!(x::Vector, storage::Vector)
        Calculus.finite_difference!(f, x, storage, :central)
        return
    end
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    function h!(x::Vector, storage::Matrix)
        Calculus.finite_difference_hessian!(f, x, storage)
        return
    end
    return TwiceDifferentiableFunction(f, g!, fg!, h!)
end

function TwiceDifferentiableFunction(f::Function,
                                     g!::Function,
                                     h!::Function)
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    return TwiceDifferentiableFunction(f, g!, fg!, h!)
end

# A cache for results from line search methods (to avoid recomputation)
type LineSearchResults{T}
    alpha::Vector{T}
    value::Vector{T}
    slope::Vector{T}
    nfailures::Int
end

LineSearchResults{T}(::Type{T}) = LineSearchResults(T[], T[], T[], 0)

Base.length(lsr::LineSearchResults) = length(lsr.alpha)

function Base.push!{T}(lsr::LineSearchResults{T}, a::T, v::T, d::T)
    push!(lsr.alpha, a)
    push!(lsr.value, v)
    push!(lsr.slope, d)
    return
end

function clear!(lsr::LineSearchResults)
    empty!(lsr.alpha)
    empty!(lsr.value)
    empty!(lsr.slope)
    return
    # nfailures is deliberately not set to 0
end
