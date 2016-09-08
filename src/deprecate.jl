@deprecate NelderMead(a::Real, g::Real, b::Real) NelderMead(initial_simplex = AffineSimplexer(), parameters = FixedParameters(a, g, b, 0.5))

@deprecate MultivariateOptimizationResults(method,
    initial_x, minimizer, minimum, iterations,
    iteration_converged, x_converged, x_tol,
    f_converged, f_tol, g_converged, g_tol,
    trace, f_calls, g_calls)  MultivariateOptimizationResults(method,
        initial_x, minimizer, minimum, iterations,
        iteration_converged, x_converged, x_tol,
        f_converged, f_tol, g_converged, g_tol,
        trace, f_calls, g_calls, 0)

# LineSearches deprecation
for name in names(LineSearches)
    if name == :LineSearches
        continue
    end
    eval(:(@deprecate $name LineSearches.$name))
end
