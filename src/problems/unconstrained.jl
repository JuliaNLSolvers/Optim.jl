module UnconstrainedProblems

### Sources
###
### [1] Ali, Khompatraporn, & Zabinsky: A Numerical Evaluation of Several Stochastic Algorithms on Selected Continuous Global Optimization Test
### Link: www.researchgate.net/profile/Montaz_Ali/publication/226654862_A_Numerical_Evaluation_of_Several_Stochastic_Algorithms_on_Selected_Continuous_Global_Optimization_Test_Problems/links/00b4952bef133a1a6b000000.pdf
###
### [2] Fletcher & Powell: A rapidly convergent descent method for minimization,

immutable OptimizationProblem
    name::AbstractString
    f::Function
    g!::Function
    h!::Function
    initial_x::Vector{Float64}
    solutions::Vector
    isdifferentiable::Bool
    istwicedifferentiable::Bool
end

examples = Dict{AbstractString, OptimizationProblem}()

##########################################################################
###
### Exponential Function
###
##########################################################################

function exponential(x::Vector)
    return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
end

function exponential_gradient!(storage::Vector, x::Vector)
    storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
    storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
end

function exponential_hessian!(storage::Matrix, x::Vector)
    storage[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
    storage[1, 2] = 0.0
    storage[2, 1] = 0.0
    storage[2, 2] = 2.0 * exp((3.0 - x[1])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
end

examples["Exponential"] = OptimizationProblem("Exponential",
                                              exponential,
                                              exponential_gradient!,
                                              exponential_hessian!,
                                              [0.0, 0.0],
                                              [2.0, 3.0],
                                              true,
                                              true)

##########################################################################
###
### Fletcher-Powell
###
### From [2]
### Source: A rapidly convergent descent method for minimization
###         Fletcher & Powell
##########################################################################

function fletcher_powell(x::Vector)
    function theta(x::Vector)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    return 100.0 * (x[3] - 10.0 * theta(x))^2 +
            (sqrt(x[1]^2 + x[2]^2) - 1.0)^2 + x[3]^2
end

# TODO: Implement
function fletcher_powell_gradient!(storage::Vector, x::Vector)
    return
end

# TODO: Implement
function fletcher_powell_hessian!(storage::Matrix, x::Vector)
    return
end

examples["Fletcher-Powell"] = OptimizationProblem("Fletcher-Powell",
                                                  fletcher_powell,
                                                  fletcher_powell_gradient!,
                                                  fletcher_powell_hessian!,
                                                  [-1.0, 0.0, 0.0], # Same as in source
                                                  [1.0, 0.0, 0.0],
                                                  false,
                                                  false)

##########################################################################
###
### Himmelblau's Function
###
##########################################################################

function himmelblau(x::Vector)
    return (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end

function himmelblau_gradient!(storage::Vector, x::Vector)
    storage[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
                  44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    storage[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
                  4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
end

function himmelblau_hessian!(storage::Matrix, x::Vector)
    storage[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
    storage[1, 2] = 4.0 * x[1] + 4.0 * x[2]
    storage[2, 1] = 4.0 * x[1] + 4.0 * x[2]
    storage[2, 2] = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
end

examples["Himmelblau"] = OptimizationProblem("Himmelblau",
                                             himmelblau,
                                             himmelblau_gradient!,
                                             himmelblau_hessian!,
                                             [2.0, 2.0],
                                             [3.0, 2.0],
                                             true,
                                             true)
##########################################################################
###
### Hosaki's Problem
###
### Problem 20 in [1]
##########################################################################

function hosaki(x::Vector)
    a = (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4)
    return a * x[2]^2 * exp(-x[2])
end

function hosaki_gradient!(storage::Vector, x::Vector)
    storage[1] = (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8)* x[2]^2 * exp(-x[2])
    storage[2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) - (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
end

function hosaki_hessian!(storage::Matrix, x::Vector)
    storage[1, 1] = (3.0 * x[1]^2 - 14.0 * x[1] + 14.0) * x[2]^2 * exp(-x[2])
    storage[1, 2] = 2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    storage[2, 1] =  2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    storage[2, 2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * exp(-x[2]) - 4.0 * ( 1.0 - 8.0 * x[1] + 7.0 *  x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) + (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
end

examples["Hosaki"] = OptimizationProblem("Hosaki",
                                         hosaki,
                                         hosaki_gradient!,
                                         hosaki_hessian!,
                                         [3.6, 1.9],
                                         [4.0, 2.0],
                                         true,
                                         true)

##########################################################################
###
### Large-Scale Quadratic
###
##########################################################################

function large_polynomial(x::Vector)
    res = zero(x[1])
    for i in 1:250
        res += (i - x[i])^2
    end
    return res
end

function large_polynomial_gradient!(storage::Vector, x::Vector)
    for i in 1:250
        storage[i] = -2.0 * (i - x[i])
    end
end

function large_polynomial_hessian!(storage::Matrix, x::Vector)
    for i in 1:250
        for j in i:250
            if i == j
                storage[i, j] = 2.0
            else
                storage[i, j] = 0.0
                storage[j, i] = 0.0
            end
        end
    end
end

examples["Large Polynomial"] = OptimizationProblem("Large Polynomial",
                                                   large_polynomial,
                                                   large_polynomial_gradient!,
                                                   large_polynomial_hessian!,
                                                   zeros(250),
                                                   float([1:250;]),
                                                   true,
                                                   true)

##########################################################################
###
### Parabola
###
##########################################################################

function parabola(x::Vector)
    return (1.0 - x[1])^2 + (2.0 - x[2])^2 + (3.0 - x[3])^2 +
            (5.0 - x[4])^2 + (8.0 - x[5])^2
end

function parabola_gradient!(storage::Vector, x::Vector)
    storage[1] = -2.0 * (1.0 - x[1])
    storage[2] = -2.0 * (2.0 - x[2])
    storage[3] = -2.0 * (3.0 - x[3])
    storage[4] = -2.0 * (5.0 - x[4])
    storage[5] = -2.0 * (8.0 - x[5])
end

function parabola_hessian!(storage::Matrix, x::Vector)
    for i in 1:5
        for j in 1:5
            if i == j
                storage[i, j] = 2.0
            else
                storage[i, j] = 0.0
            end
        end
    end
end

examples["Parabola"] = OptimizationProblem("Parabola",
                                           parabola,
                                           parabola_gradient!,
                                           parabola_hessian!,
                                           [0.0, 0.0, 0.0, 0.0, 0.0],
                                           [1.0, 2.0, 3.0, 5.0, 8.0],
                                           true,
                                           true)

##########################################################################
###
### Simple 4th-Degree Polynomial Example
###
##########################################################################

function polynomial(x::Vector)
    return (10.0 - x[1])^2 + (7.0 - x[2])^4 + (108.0 - x[3])^4
end

function polynomial_gradient!(storage::Vector, x::Vector)
    storage[1] = -2.0 * (10.0 - x[1])
    storage[2] = -4.0 * (7.0 - x[2])^3
    storage[3] = -4.0 * (108.0 - x[3])^3
end

function polynomial_hessian!(storage::Matrix, x::Vector)
    storage[1, 1] = 2.0
    storage[1, 2] = 0.0
    storage[1, 3] = 0.0
    storage[2, 1] = 0.0
    storage[2, 2] = 12.0 * (7.0 - x[2])^2
    storage[2, 3] = 0.0
    storage[3, 1] = 0.0
    storage[3, 2] = 0.0
    storage[3, 3] = 12.0 * (108.0 - x[3])^2
end

examples["Polynomial"] = OptimizationProblem("Polynomial",
                                             polynomial,
                                             polynomial_gradient!,
                                             polynomial_hessian!,
                                             [0.0, 0.0, 0.0],
                                             [10.0, 7.0, 108.0],
                                             true,
                                             true)

##########################################################################
###
### Powell (d=4)
###
### Problem 35 in [1]
### Difficult since the hessian is singular at the optimum
##########################################################################

function powell(x::Vector)
    return (x[1] + 10.0 * x[2])^2 + 5.0 * (x[3] - x[4])^2 +
            (x[2] - 2.0 * x[3])^4 + 10.0 * (x[1] - x[4])^4
end

function powell_gradient!(storage::Vector, x::Vector)
    storage[1] = 2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3
    storage[2] = 20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3
    storage[3] = 10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3
    storage[4] = -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3
end

function powell_hessian!(storage::Matrix, x::Vector)
    storage[1, 1] = 2.0 + 120.0 * (x[1] - x[4])^2
    storage[1, 2] = 20.0
    storage[1, 3] = 0.0
    storage[1, 4] = -120.0 * (x[1] - x[4])^2
    storage[2, 1] = 20.0
    storage[2, 2] = 200.0 + 12.0 * (x[2] - 2.0 * x[3])^2
    storage[2, 3] = -24.0 * (x[2] - 2.0 * x[3])^2
    storage[2, 4] = 0.0
    storage[3, 1] = 0.0
    storage[3, 2] = -24.0 * (x[2] - 2.0 * x[3])^2
    storage[3, 3] = 10.0 + 48.0 * (x[2] - 2.0 * x[3])^2
    storage[3, 4] = -10.0
    storage[4, 1] = -120.0 * (x[1] - x[4])^2
    storage[4, 2] = 0.0
    storage[4, 3] = -10.0
    storage[4, 4] = 10.0 + 120.0 * (x[1] - x[4])^2
end

examples["Powell"] = OptimizationProblem("Powell",
                                         powell,
                                         powell_gradient!,
                                         powell_hessian!,
                                         [3.0, -1.0, 0.0, 1.0],
                                         [0.0, 0.0, 0.0, 0.0],
                                         true,
                                         true)

##########################################################################
###
### Rosenbrock (2D)
###
### Problem 38 in [1]
###
### Saddle point makes optimization difficult
##########################################################################

function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient!(storage::Vector, x::Vector)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

function rosenbrock_hessian!(storage::Matrix, x::Vector)
    storage[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    storage[1, 2] = -400.0 * x[1]
    storage[2, 1] = -400.0 * x[1]
    storage[2, 2] = 200.0
end

examples["Rosenbrock"] = OptimizationProblem("Rosenbrock",
                                             rosenbrock,
                                             rosenbrock_gradient!,
                                             rosenbrock_hessian!,
                                             [0.0, 0.0],
                                             [1.0, 1.0],
                                             true,
                                             true)

end # module
