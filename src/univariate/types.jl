"""
    UnivariateOptimizationResults

Concrete result container for bounded one-dimensional optimization methods.

# Fields

- `method`: The optimizer instance that produced the result.
- `initial_lower`, `initial_upper`: The initial search interval.
- `minimizer`: The final candidate point.
- `minimum`: The objective value at `minimizer`.
- `iterations`: Number of completed optimization iterations.
- `rel_tol`, `abs_tol`: Relative and absolute interval tolerances used for convergence.
- `trace`: Stored optimization trace.
- `f_calls`: Number of objective evaluations.
- `time_limit`: Maximum allowed runtime in seconds, or `NaN` when unlimited.
- `time_run`: Runtime in seconds.
- `stopped_by`: Named tuple containing the termination flags.

Use the generic result accessors rather than depending on this concrete field
layout.
"""
mutable struct UnivariateOptimizationResults{Tb,Tt,Tf,Tx,M,O<:UnivariateOptimizer,Tsb<:NamedTuple} <:
               OptimizationResults
    method::O
    initial_lower::Tb
    initial_upper::Tb
    minimizer::Tx
    minimum::Tf
    iterations::Int
    rel_tol::Tt
    abs_tol::Tt
    trace::OptimizationTrace{M}
    f_calls::Int
    time_limit::Float64
    time_run::Float64
    stopped_by::Tsb
end
