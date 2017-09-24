module ConstrainedProblems

using Optim, Compat

immutable ConstrainedProblem{C<:Optim.AbstractConstraints}
    name::String
    f::Function
    g!::Function
    h!::Function
    constraints::C
    initial_x::Vector{Float64}
    solutions::Vector
    isdifferentiable::Bool
    istwicedifferentiable::Bool
end

pexamples = Dict{String, ConstrainedProblem}()

#####################################
###
### Minimum distance, box constraints
###
#####################################

function sqrdist(x)
    dx1 = x[1] - 5
    dx2 = x[2] - 3
    (dx1*dx1 + dx2*dx2)/2
end

function sqrdist_gradient!(x, g)
    g[1] = x[1] - 5
    g[2] = x[2] - 3
end

function sqrdist_hessian!(x, H)
    H[1,1] = H[2,2] = 1
    H[1,2] = H[2,1] = 0
end

constraints = Optim.ConstraintsBox([-2.0,-2.0],[2.0,2.0])

pexamples["SqrdistBox"] = ConstrainedProblem("SqrdistBox",
                                            sqrdist,
                                            sqrdist_gradient!,
                                            sqrdist_hessian!,
                                            constraints,
                                            [0.0, 0.0],
                                            Any[[2.0, 2.0]],
                                            true,
                                            true)

constraints = Optim.ConstraintsBox(nothing,[2.0,2.0])

pexamples["SqrdistLwr"] = ConstrainedProblem("SqrdistLwr",
                                            sqrdist,
                                            sqrdist_gradient!,
                                            sqrdist_hessian!,
                                            constraints,
                                            [0.0, 0.0],
                                            Any[[2.0, 2.0]],
                                            true,
                                            true)

constraints = Optim.ConstraintsBox([-2.0,-2.0],nothing)

pexamples["SqrdistLwr"] = ConstrainedProblem("SqrdistLwr",
                                            sqrdist,
                                            sqrdist_gradient!,
                                            sqrdist_hessian!,
                                            constraints,
                                            [0.0, 0.0],
                                            Any[[5.0, 3.0]],
                                            true,
                                            true)

end # module
