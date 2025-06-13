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
