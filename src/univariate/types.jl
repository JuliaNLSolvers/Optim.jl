mutable struct UnivariateOptimizationResults{Tb,Tt,Tf, Tx,M,O<:UnivariateOptimizer} <: OptimizationResults
    method::O
    initial_lower::Tb
    initial_upper::Tb
    minimizer::Tx
    minimum::Tf
    iterations::Int
    iteration_converged::Bool
    converged::Bool
    rel_tol::Tt
    abs_tol::Tt
    trace::OptimizationTrace{M}
    f_calls::Int
end
