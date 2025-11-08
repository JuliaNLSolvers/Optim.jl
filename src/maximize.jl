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
    fmax = let f=f
        x -> -f(x)
    end
    MaximizationWrapper(optimize(fmax, lb, ub, method; kwargs...))
end

function maximize(f, lb::Real, ub::Real; kwargs...)
    fmax = let f=f
        x -> -f(x)
    end
    MaximizationWrapper(optimize(fmax, lb, ub; kwargs...))
end


# ==============================================================================
# Multivariate warppers
# ==============================================================================
function maximize(f, x0::AbstractArray, options::Options = Options())
    fmax = let f=f
        x -> -f(x)
    end
    MaximizationWrapper(optimize(fmax, x0, options))
end
function maximize(
    f,
    x0::AbstractArray,
    method::AbstractOptimizer,
    options = Options()
)
    fmax = let f=f
        x -> -f(x)
    end
    MaximizationWrapper(optimize(fmax, x0, method, options))
end
function maximize(
    f,
    g,
    x0::AbstractArray,
    method::AbstractOptimizer,
    options = Options()
)
    fmax = let f=f
        x -> -f(x)
    end
    gmax = let g=g
        (G, x) -> (g(G, x); G .= .-G)
    end
    MaximizationWrapper(optimize(fmax, gmax, x0, method, options))
end

function maximize(
    f,
    g,
    h,
    x0::AbstractArray,
    method::AbstractOptimizer,
    options = Options()
)
    fmax = let f=f
        x -> -f(x)
    end
    gmax = let g=g
        (G, x) -> (g(G, x); G .= .-G)
    end
    hmax = let h=h
        (H, x) -> (h(H, x); H .= .-H)
    end
    MaximizationWrapper(optimize(fmax, gmax, hmax, x0, method, options))
end

maximizer(r::MaximizationWrapper) = minimizer(res(r))
Base.maximum(r::MaximizationWrapper) = -minimum(res(r))
Base.summary(io::IO, r::MaximizationWrapper) = summary(io, res(r))

for api_method in (
    :lower_bound,
    :upper_bound,
    :rel_tol,
    :abs_tol,
    :iterations,
    :initial_state,
    :converged,
    :x_tol,
    :x_abstol,
    :x_reltol,
    :x_converged,
    :x_abschange,
    :g_tol,
    :g_abstol,
    :g_converged,
    :g_residual,
    :f_tol,
    :f_reltol,
    :f_abstol,
    :f_converged,
    :f_increased,
    :f_relchange,
    :iteration_limit_reached,
    :f_calls,
    :g_calls,
    :h_calls,
)
    @eval $api_method(r::MaximizationWrapper) = $api_method(res(r))
end

function Base.show(io::IO, r::MaximizationWrapper{<:UnivariateOptimizationResults})
    println(io, "Results of Maximization Algorithm")
    print(io, " * Algorithm: ")
    summary(io, r)
    println(io)
    @printf io " * Search Interval: [%f, %f]\n" lower_bound(r) upper_bound(r)
    @printf io " * Maximizer: %e\n" maximizer(r)
    @printf io " * Maximum: %e\n" maximum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(%.1e*|x|+%.1e): %s\n" rel_tol(
        r,
    ) abs_tol(r) converged(r)
    @printf io " * Objective Function Calls: %d" f_calls(r)
    return
end


Base.show(io::IO, r::MaximizationWrapper{<:MultivariateOptimizationResults}) = show(io, r.res)
