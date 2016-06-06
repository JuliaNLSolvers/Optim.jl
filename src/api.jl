method(r::OptimizationResults) = r.method
minimizer(r::OptimizationResults) = r.minimum
minimum(r::OptimizationResults) = r.f_minimum
iterations(r::OptimizationResults) = r.iterations
iteration_limit_reached(r::OptimizationResults) = r.iteration_converged
trace(r::OptimizationResults) = length(r.trace.states) > 0 ? r.trace : error("No trace in optimization results. To get a trace, run optimize() with store_trace = true.")

function x_trace(r::UnivariateOptimizationResults)
	tr = trace(r)
	!haskey(tr.states[1].metadata, "x_minimum") && error("Trace does not contain x. To get a trace of x, run optimize() with extended_trace = true")
    [state.metadata["x_minimum"] for state in tr.states]
end
function x_lower_trace(r::UnivariateOptimizationResults)
	tr = trace(r)
	!haskey(tr.states[1].metadata, "x_lower") && error("Trace does not contain x. To get a trace of x, run optimize() with extended_trace = true")
    [state.metadata["x_lower"] for state in tr.states]
end
x_lower_trace(r::MultivariateOptimizationResults) = error("x_lower_trace is not implemented for $(method(r)).")
function x_upper_trace(r::UnivariateOptimizationResults)
	tr = trace(r)
	!haskey(tr.states[1].metadata, "x_upper") && error("Trace does not contain x. To get a trace of x, run optimize() with extended_trace = true")
    [state.metadata["x_upper"] for state in tr.states]
end
x_upper_trace(r::MultivariateOptimizationResults) = error("x_upper_trace is not implemented for $(method(r)).")

function x_trace(r::MultivariateOptimizationResults)
	tr = trace(r)
	!haskey(tr.states[1].metadata, "x") && error("Trace does not contain x. To get a trace of x, run optimize() with extended_trace = true")
    [state.metadata["x"] for state in tr.states]
end

f_trace(r::OptimizationResults) = [state.value for state in trace(r).states]
g_norm_trace(r::OptimizationResults) = error("g_norm_trace is not implemented for $(method(r)).")
g_norm_trace(r::MultivariateOptimizationResults) = [state.g_norm for state in trace(r).states]

f_calls(r::OptimizationResults) = r.f_calls

g_calls(r::OptimizationResults) = error("g_calls is not implemented for $(method(r)).")
g_calls(r::MultivariateOptimizationResults) = r.g_calls

converged(r::UnivariateOptimizationResults) = r.converged
converged(r::MultivariateOptimizationResults) = r.x_converged || r.f_converged || r.g_converged
x_converged(r::OptimizationResults) = error("x_converged is not implemented for $(method(r)).")
x_converged(r::MultivariateOptimizationResults) = r.x_converged
f_converged(r::OptimizationResults) = error("f_converged is not implemented for $(method(r)).")
f_converged(r::MultivariateOptimizationResults) = r.f_converged
g_converged(r::OptimizationResults) = error("g_converged is not implemented for $(method(r)).")
g_converged(r::MultivariateOptimizationResults) = r.g_converged

x_tol(r::OptimizationResults) = error("x_tol is not implemented for $(method(r)).")
x_tol(r::MultivariateOptimizationResults) = r.x_tol
f_tol(r::OptimizationResults) = error("f_tol is not implemented for $(method(r)).")
f_tol(r::MultivariateOptimizationResults) = r.f_tol
g_tol(r::OptimizationResults) = error("g_tol is not implemented for $(method(r)).")
g_tol(r::MultivariateOptimizationResults) = r.g_tol


initial_state(r::OptimizationResults) = error("initial_state is not implemented for $(method(r)).")
initial_state(r::MultivariateOptimizationResults) = r.initial_x

lower_bound(r::OptimizationResults) = error("lower_bound is not implemented for $(method(r)).")
lower_bound(r::UnivariateOptimizationResults) = r.initial_lower
upper_bound(r::OptimizationResults) = error("upper_bound is not implemented for $(method(r)).")
upper_bound(r::UnivariateOptimizationResults) = r.initial_upper

rel_tol(r::OptimizationResults) = error("rel_tol is not implemented for $(method(r)).")
rel_tol(r::UnivariateOptimizationResults) = r.rel_tol
abs_tol(r::OptimizationResults) = error("abs_tol is not implemented for $(method(r)).")
abs_tol(r::UnivariateOptimizationResults) = r.abs_tol
