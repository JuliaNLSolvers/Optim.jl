Base.summary(r::OptimizationResults) = summary(r.method) # might want to do more here than just return summary of the method used
minimizer(r::OptimizationResults) = r.minimizer
minimum(r::OptimizationResults) = r.minimum
iterations(r::OptimizationResults) = r.iterations
iteration_limit_reached(r::OptimizationResults) = r.iteration_converged
trace(r::OptimizationResults) = length(r.trace) > 0 ? r.trace : error("No trace in optimization results. To get a trace, run optimize() with store_trace = true.")

function x_trace(r::UnivariateOptimizationResults)
    tr = trace(r)
    !haskey(tr[1].metadata, "minimizer") && error("Trace does not contain x. To get a trace of x, run optimize() with extended_trace = true")
    [ state.metadata["minimizer"] for state in tr ]
end
function x_lower_trace(r::UnivariateOptimizationResults)
    tr = trace(r)
    !haskey(tr[1].metadata, "x_lower") && error("Trace does not contain x. To get a trace of x, run optimize() with extended_trace = true")
    [ state.metadata["x_lower"] for state in tr ]
end
x_lower_trace(r::MultivariateOptimizationResults) = error("x_lower_trace is not implemented for $(summary(r)).")
function x_upper_trace(r::UnivariateOptimizationResults)
    tr = trace(r)
    !haskey(tr[1].metadata, "x_upper") && error("Trace does not contain x. To get a trace of x, run optimize() with extended_trace = true")
    [ state.metadata["x_upper"] for state in tr ]
end
x_upper_trace(r::MultivariateOptimizationResults) = error("x_upper_trace is not implemented for $(summary(r)).")

function x_trace(r::MultivariateOptimizationResults)
    tr = trace(r)
    !haskey(tr[1].metadata, "x") && error("Trace does not contain x. To get a trace of x, run optimize() with extended_trace = true")
    [ state.metadata["x"] for state in tr ]
end

f_trace(r::OptimizationResults) = [ state.value for state in trace(r) ]
g_norm_trace(r::OptimizationResults) = error("g_norm_trace is not implemented for $(summary(r)).")
g_norm_trace(r::MultivariateOptimizationResults) = [ state.g_norm for state in trace(r) ]

f_calls(r::OptimizationResults) = r.f_calls
f_calls(d) = first(d.f_calls)

g_calls(r::OptimizationResults) = error("g_calls is not implemented for $(summary(r)).")
g_calls(r::MultivariateOptimizationResults) = r.g_calls
g_calls(d::NonDifferentiable) = 0
g_calls(d) = first(d.df_calls)

h_calls(r::OptimizationResults) = error("h_calls is not implemented for $(summary(r)).")
h_calls(r::MultivariateOptimizationResults) = r.h_calls
h_calls(d::Union{NonDifferentiable, OnceDifferentiable}) = 0
h_calls(d) = first(d.h_calls)
h_calls(d::TwiceDifferentiableHV) = first(d.hv_calls)

converged(r::UnivariateOptimizationResults) = r.converged
converged(r::MultivariateOptimizationResults) = r.x_converged || r.f_converged || r.g_converged
x_converged(r::OptimizationResults) = error("x_converged is not implemented for $(summary(r)).")
x_converged(r::MultivariateOptimizationResults) = r.x_converged
f_converged(r::OptimizationResults) = error("f_converged is not implemented for $(summary(r)).")
f_converged(r::MultivariateOptimizationResults) = r.f_converged
f_increased(r::OptimizationResults) = error("f_increased is not implemented for $(summary(r)).")
f_increased(r::MultivariateOptimizationResults) = r.f_increased
g_converged(r::OptimizationResults) = error("g_converged is not implemented for $(summary(r)).")
g_converged(r::MultivariateOptimizationResults) = r.g_converged

x_tol(r::OptimizationResults) = error("x_tol is not implemented for $(summary(r)).")
x_tol(r::MultivariateOptimizationResults) = r.x_tol
x_abschange(r::MultivariateOptimizationResults) = r.x_abschange
f_tol(r::OptimizationResults) = error("f_tol is not implemented for $(summary(r)).")
f_tol(r::MultivariateOptimizationResults) = r.f_tol
f_abschange(r::MultivariateOptimizationResults) = r.f_abschange
@inline function f_relchange(r::MultivariateOptimizationResults)
    fabs = f_abschange(r)
    if fabs == zero(fabs)
        return zero(fabs)
    else
        return fabs / r.minimum # TODO: wrong value if f_increased is true
    end
end
g_tol(r::OptimizationResults) = error("g_tol is not implemented for $(summary(r)).")
g_tol(r::MultivariateOptimizationResults) = r.g_tol
g_residual(r::MultivariateOptimizationResults) = r.g_residual


initial_state(r::OptimizationResults) = error("initial_state is not implemented for $(summary(r)).")
initial_state(r::MultivariateOptimizationResults) = r.initial_x

lower_bound(r::OptimizationResults) = error("lower_bound is not implemented for $(summary(r)).")
lower_bound(r::UnivariateOptimizationResults) = r.initial_lower
upper_bound(r::OptimizationResults) = error("upper_bound is not implemented for $(summary(r)).")
upper_bound(r::UnivariateOptimizationResults) = r.initial_upper

rel_tol(r::OptimizationResults) = error("rel_tol is not implemented for $(summary(r)).")
rel_tol(r::UnivariateOptimizationResults) = r.rel_tol
abs_tol(r::OptimizationResults) = error("abs_tol is not implemented for $(summary(r)).")
abs_tol(r::UnivariateOptimizationResults) = r.abs_tol
