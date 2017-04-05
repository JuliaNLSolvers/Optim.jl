type UnivariateOptimizationResults{T,M} <: OptimizationResults
    method::String
    initial_lower::T
    initial_upper::T
    minimizer::T
    minimum::T
    iterations::Int
    iteration_converged::Bool
    converged::Bool
    rel_tol::T
    abs_tol::T
    trace::OptimizationTrace{M}
    f_calls::Int
end
