abstract type ConstrainedOptimizer{T}  <: AbstractOptimizer end
abstract type IPOptimizer{T} <: ConstrainedOptimizer{T} end # interior point methods
