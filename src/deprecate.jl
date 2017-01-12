const has_deprecated_linesearch! = Ref(false)
const has_deprecated_precondprep! = Ref(false)
const has_deprecated_levenberg_marquardt = Ref(false)

@deprecate MultivariateOptimizationResults(method,
    initial_x, minimizer, minimum, iterations,
    iteration_converged, x_converged, x_tol,
    f_converged, f_tol, g_converged, g_tol,
    trace, f_calls, g_calls)  MultivariateOptimizationResults(method,
        initial_x, minimizer, minimum, iterations,
        iteration_converged, x_converged, x_tol,
        f_converged, f_tol, g_converged, g_tol, false,
        trace, f_calls, g_calls, 0)

@deprecate MultivariateOptimizationResults(method,
    initial_x, minimizer, minimum, iterations,
    iteration_converged, x_converged, x_tol,
    f_converged, f_tol, g_converged, g_tol,
    trace, f_calls, g_calls, h_calls)  MultivariateOptimizationResults(method,
        initial_x, minimizer, minimum, iterations,
        iteration_converged, x_converged, x_tol,
        f_converged, f_tol, g_converged, g_tol, false,
        trace, f_calls, g_calls, h_calls)

# LineSearches deprecation
for name in names(LineSearches)
    if name in (:LineSearches, :hz_linesearch!, :mt_linesearch!,
                :interpolating_linesearch!, :backtracking_linesearch!,
                :interpbacktracking_linesearch!)
        continue
    end
    eval(:(@deprecate $name LineSearches.$name))
end

@deprecate hz_linesearch! LineSearches.hagerzhang!
@deprecate mt_linesearch! LineSearches.morethuente!
@deprecate interpolating_linesearch! LineSearches.strongwolfe!
@deprecate backtracking_linesearch! LineSearches.backtracking!
@deprecate interpbacktracking_linesearch! LineSearches.interpbacktracking!

@deprecate OptimizationOptions(args...; kwargs...) Optim.Options(args...; kwargs...)

function get_linesearch(linesearch!, linesearch)
    if linesearch! != nothing
        if !has_deprecated_linesearch![]
            warn("linesearch! keyword is deprecated, use linesearch instead (without !)")
            has_deprecated_linesearch![] = true
        end
        return linesearch!
    end
    linesearch
end

function get_precondprep(precondprep!, precondprep)
    if precondprep! != nothing
        if !has_deprecated_precondprep![]
            warn("precondprep! keyword is deprecated, use precondprep instead (without !)")
            has_deprecated_precondprep![] = true
        end
        return precondprep!
    end
    precondprep
end
