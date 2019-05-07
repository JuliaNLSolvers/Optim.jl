# In v1.0 we should possibly just overload getproperty
# Base.getproperty(w::WrapVar, s::Symbol) = getfield(w.v, s)
struct MaximizationWrapper{T}
    res::T
end
res(r::MaximizationWrapper) = r.res

# ==============================================================================
# Univariate warppers
# ==============================================================================
function maximize(f, lb::Real, ub::Real, method::AbstractOptimizer; kwargs...)
    fmax = x->-f(x)
    MaximizationWrapper(optimize(fmax, lb, ub, method; kwargs...))
end

function maximize(f, lb::Real, ub::Real; kwargs...)
    fmax = x->-f(x)
    MaximizationWrapper(optimize(fmax, lb, ub; kwargs...))
end

# ==============================================================================
# Multivariate warppers
# ==============================================================================
function maximize(f, x0::AbstractArray; kwargs...)
    fmax = x->-f(x)
    MaximizationWrapper(optimize(fmax, x0; kwargs...))
end
function maximize(f, x0::AbstractArray, method::AbstractOptimizer, options = Optim.Options(); kwargs...)
    fmax = x->-f(x)
    MaximizationWrapper(optimize(fmax, x0, method, options; kwargs...))
end
function maximize(f, g, x0::AbstractArray, method::AbstractOptimizer, options = Optim.Options(); kwargs...)
    fmax = x->-f(x)
    gmax = (G,x)->(g(G,x); G.=-G)
    MaximizationWrapper(optimize(fmax, gmax, x0, method, options; kwargs...))
end

function maximize(f, g, h, x0::AbstractArray, method::AbstractOptimizer, options = Optim.Options(); kwargs...)
    fmax = x->-f(x)
    gmax = (G,x)->(g(G,x); G.=-G)
    hmax = (H,x)->(h(G,x); H.=-H)
    MaximizationWrapper(optimize(fmax, gmax, hmax, x0, method, options; kwargs...))
end

minimum(r::MaximizationWrapper) = throw(MethodError())
maximizer(r::Union{UnivariateOptimizationResults,MultivariateOptimizationResults}) = throw(MethodError())
maximizer(r::MaximizationWrapper) = minimizer(res(r))
maximum(r::Union{UnivariateOptimizationResults,MultivariateOptimizationResults}) = throw(MethodError())
maximum(r::MaximizationWrapper) = -minimum(res(r))
Base.summary(r::MaximizationWrapper) = summary(res(r))

for api_method in (:lower_bound, :upper_bound, :rel_tol, :abs_tol, :iterations, :initial_state, :converged, :x_tol, :x_converged,
               :x_abschange, :g_tol, :g_converged, :g_residual, :f_tol, :f_converged,
               :f_increased, :f_relchange, :iteration_limit_reached, :f_calls,
               :g_calls, :h_calls)
   @eval $api_method(r::MaximizationWrapper) = $api_method(res(r))
end

function Base.show(io::IO, r::MaximizationWrapper{<:UnivariateOptimizationResults})
    @printf io "Results of Maximization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r)
    @printf io " * Search Interval: [%f, %f]\n" lower_bound(r) upper_bound(r)
    @printf io " * Maximizer: %e\n" maximizer(r)
    @printf io " * Maximum: %e\n" maximum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(%.1e*|x|+%.1e): %s\n" rel_tol(r) abs_tol(r) converged(r)
    @printf io " * Objective Function Calls: %d" f_calls(r)
    return
end

function Base.show(io::IO, r::MaximizationWrapper{<:MultivariateOptimizationResults})
    take = Iterators.take

    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r.res)
    if length(join(initial_state(r), ",")) < 40
        @printf io " * Starting Point: [%s]\n" join(initial_state(r), ",")
    else
        @printf io " * Starting Point: [%s, ...]\n" join(take(initial_state(r),
2), ",")
    end
    if length(join(maximizer(r), ",")) < 40
        @printf io " * Maximizer: [%s]\n" join(maximizer(r), ",")
    else
        @printf io " * Maximizer: [%s, ...]\n" join(take(maximizer(r), 2), ",")
    end
    @printf io " * Maximum: %e\n" maximum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: %s\n" converged(r)
    if isa(r.res.method, NelderMead)
        @printf io "   *  √(Σ(yᵢ-ȳ)²)/n < %.1e: %s\n" g_tol(r) g_converged(r)
    else
        @printf io "   * |x - x'| ≤ %.1e: %s \n" x_tol(r) x_converged(r)
        @printf io "     |x - x'| = %.2e \n"  x_abschange(r)
        @printf io "   * |f(x) - f(x')| ≤ %.1e |f(x)|: %s\n" f_tol(r) f_converged(r)
        @printf io "     |f(x) - f(x')| = %.2e |f(x)|\n" f_relchange(r)
        @printf io "   * |g(x)| ≤ %.1e: %s \n" g_tol(r) g_converged(r)
        @printf io "     |g(x)| = %.2e \n"  g_residual(r)
        @printf io "   * Stopped by an decreasing objective: %s\n" (f_increased(r) && !iteration_limit_reached(r))
    end
    @printf io "   * Reached Maximum Number of Iterations: %s\n" iteration_limit_reached(r)
    @printf io " * Objective Calls: %d" f_calls(r)
    if !(isa(r.res.method, NelderMead) || isa(r.res.method, SimulatedAnnealing))
        @printf io "\n * Gradient Calls: %d" g_calls(r)
    end
    if isa(r.res.method, Newton) || isa(r.res.method, NewtonTrustRegion)
        @printf io "\n * Hessian Calls: %d" h_calls(r)
    end
    return
end
