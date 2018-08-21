# In v1.0 we should possibly just overload getproperty
# Base.getproperty(w::WrapVar, s::Symbol) = getfield(w.v, s)
struct MaxWrap{T}
    res::T
end

function maximize(f, x0::AbstractArray, method::AbstractOptimizer, options = Optim.Options(); kwargs...)
    fmax = x->-f(x)
    MaxWrap(optimize(fmax, x0, method, options; kwargs...))
end
function maximize(f, g, x0::AbstractArray, method::AbstractOptimizer, options = Optim.Options(); kwargs...)
    fmax = x->-f(x)
    gmax = (G,x)->(g(G,x); G.=-G)
    MaxWrap(optimize(fmax, gmax, x0, method, options; kwargs...))
end

function maximize(f, g, h, x0::AbstractArray, method::AbstractOptimizer, options = Optim.Options(); kwargs...)
    fmax = x->-f(x)
    gmax = (G,x)->(g(G,x); G.=-G)
    hmax = (H,x)->(h(G,x); H.=-H)
    MaxWrap(optimize(fmax, gmax, hmax, x0, method, options; kwargs...))
end

minimum(r::MaxWrap) = throw(MethodError())
maximizer(r::Union{UnivariateOptimizationResults,MultivariateOptimizationResults}) = throw(MethodError())
maximizer(r::MaxWrap) = minimizer(r.res)
maximum(r::Union{UnivariateOptimizationResults,MultivariateOptimizationResults}) = throw(MethodError())
maximum(r::MaxWrap) = -r.res.minimum

for api_method in (:iterations, :initial_state, :converged, :x_tol, :x_converged,
               :x_abschange, :g_tol, :g_converged, :g_residual, :f_tol, :f_converged,
               :f_increased, :f_relchange, :iteration_limit_reached, :f_calls,
               :g_calls, :h_calls)
   @eval $api_method(r::MaxWrap) = $api_method(r.res)
end

function Base.show(io::IO, r::MaxWrap)
    first_two(fr) = [x for (i, x) in enumerate(fr)][1:2]

    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r.res)
    if length(join(initial_state(r), ",")) < 40
        @printf io " * Starting Point: [%s]\n" join(initial_state(r), ",")
    else
        @printf io " * Starting Point: [%s, ...]\n" join(first_two(initial_state(r)), ",")
    end
    if length(join(maximizer(r), ",")) < 40
        @printf io " * Maximizer: [%s]\n" join(maximizer(r), ",")
    else
        @printf io " * Maximizer: [%s, ...]\n" join(first_two(maximizer(r)), ",")
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
