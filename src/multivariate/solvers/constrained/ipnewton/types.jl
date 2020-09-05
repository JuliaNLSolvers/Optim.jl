abstract type ConstrainedOptimizer{T}  <: AbstractConstrainedOptimizer end
abstract type IPOptimizer{T} <: ConstrainedOptimizer{T} end # interior point methods
