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

@deprecate hz_linesearch! LineSearches.hagerzhang!
@deprecate mt_linesearch! LineSearches.morethuente!
@deprecate interpolating_linesearch! LineSearches.strongwolfe!
@deprecate backtracking_linesearch! LineSearches.backtracking!
@deprecate interpbacktracking_linesearch! LineSearches.interpbacktracking!

if VERSION >= v"0.5.0"
    view5(A, i, j) = view(A, i, j)
else
    view5(A, i, j) = A[i,j]
end
