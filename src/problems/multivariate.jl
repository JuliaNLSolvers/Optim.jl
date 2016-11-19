module MultivariateProblems

export UnconstrainedProblems, ConstrainedProblems

immutable OptimizationProblem
    name::AbstractString
    f::Function
    g!::Function
    h!::Function
    constraints
    initial_x::Vector{Float64}
    solutions::Vector
    isdifferentiable::Bool
    istwicedifferentiable::Bool
end

function OptimizationProblem(name, f, g!, h!,
                             initial_x::AbstractVector, solutions,
                             isdifferentiable::Bool, istwicedifferentiable::Bool)
    OptimizationProblem(name, f, g!, h!, nothing,
                        initial_x, solutions, isdifferentiable, istwicedifferentiable)
end

include("unconstrained.jl")
include("constrained.jl")

end
