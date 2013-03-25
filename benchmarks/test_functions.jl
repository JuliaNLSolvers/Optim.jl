##########################################################################
###
### Fletcher-Powell
###
##########################################################################

function fletcher_powell(x::Vector)
    function theta(x::Vector)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    return 100.0 * (x[3] - 10.0 * theta(x[1], x[2]))^2 +
            (sqrt(x[1]^2 + x[2]^2) - 1.0)^2 + x[3]^2
end

function fletcher_powell_gradient(x::Vector, storage::Vector)
    error("Not yet defined")
end

function fletcher_powell_hessian(x::Vector, storage::Matrix)
    error("Not yet defined")
end

##########################################################################
###
### Himmelbrau's Function
###
##########################################################################

function himmelbrau(x::Vector)
    return (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end

function himmelbrau_gradient!(x::Vector, storage::Vector)
    storage[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] +
                  44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    storage[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
                  4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
end

function himmelbrau_hessian!(x::Vector, storage::Matrix)
    error("Not yet defined")
end

##########################################################################
###
### Parabola
###
##########################################################################

function parabola(x::Vector)
    return (1.0 - x[1])^2 + (2.0 - x[2])^2 + (3.0 - x[3])^2 +
            (5.0 - x[4])^2 + (8.0 - x[5])^2
end

function parabola_gradient!(x::Vector, storage::Vector)
    storage[1] = -2.0 * (1.0 - x[1])
    storage[2] = -2.0 * (2.0 - x[2])
    storage[3] = -2.0 * (3.0 - x[3])
    storage[4] = -2.0 * (5.0 - x[4])
    storage[5] = -2.0 * (8.0 - x[5])
end

function parabola_hessian!(x::Vector, storage::Matrix)
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

##########################################################################
###
### Powell
###
##########################################################################

function powell(x::Vector)
    return (x[1] + 10.0 * x[2])^2 + 5.0 * (x[3] - x[4])^2 + 
            (x[2] - 2.0 * x[3])^4 + 10.0 * (x[1] - x[4])^4
end

function powell_gradient!(x::Vector, storage::Vector)
    storage[1] = 2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3
    storage[2] = 20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3
    storage[3] = 10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3
    storage[4] = -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3
end

function powell_hessian!(x::Vector, storage::Matrix)
    storage[1, 1] = 2.0 + 120.0 * (x[1] - x[4])^2
    storage[1, 2] = 20.0
    storage[1, 3] = 0.0
    storage[1, 4] = 0.0
    storage[2, 1] = 20.0
    storage[2, 2] = 200.0 + 12.0 * (x[2] - 2.0 * x[3])^2
    storage[2, 3] = -24.0 * (x[2] - 2.0 * x[3])^2
    storage[2, 4] = 0.0
    storage[3, 1] = 0.0
    storage[3, 2] = -24.0 * (x[2] - 2.0 * x[3])^2
    storage[3, 3] = 10.0 + 48.0 * (x[2] - 2.0 * x[3])^2
    storage[3, 4] = -10.0
    storage[4, 1] = 0.0
    storage[4, 2] = 0.0
    storage[4, 3] = -10.0
    storage[4, 4] = 10.0 + 120.0 * (x[1] - x[4])^2
end

##########################################################################
###
### Rosenbrock
###
##########################################################################

function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient!(x::Vector, storage::Vector)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

function rosenbrock_hessian!(x::Vector, storage::Matrix)
    storage[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    storage[1, 2] = -400.0 * x[1]
    storage[2, 1] = -400.0 * x[1]
    storage[2, 2] = 200.0
end

##########################################################################
###
### Simple 4th-Degree Polynomial Example
###
##########################################################################

function polynomial(x::Vector)
    return (10.0 - x[1])^2 + (7.0 - x[2])^4 + (108.0 - x[3])^4
end

function polynomial_gradient!(x::Vector, storage::Vector)
    storage[1] = -2.0 * (10.0 - x[1])
    storage[2] = -4.0 * (7.0 - x[2])^3
    storage[3] = -4.0 * (108.0 - x[3])^3
end

function polynomial_hessian!(x::Vector, storage::Matrix)
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

##########################################################################
###
### Simple Exponential Function Example
###
##########################################################################

function exponential(x::Vector)
    return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
end

function exponential_gradient!(x::Vector, storage::Vector)
    storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
    storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
end

function exponential_hessian!(x::Vector, storage::Matrix)
    storage[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
    storage[1, 2] = 0.0
    storage[2, 1] = 0.0
    storage[2, 2] = 2.0 * exp((3.0 - x[1])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
end

##########################################################################
###
### Large-Scale Quadratic
###
##########################################################################

function large_polynomial(x::Vector)
    res = 0.0
    for i in 1:250
        res += (i - x[i])^2
    end
    return res
end

function large_polynomial_gradient!(x::Vector, storage::Vector)
    for i in 1:250
        storage[i] = -2.0 * (i - x[i])
    end
end

function large_polynomial_hessian!(x::Vector, storage::Matrix)
    for i in 1:250
        for j in 1:250
            if i == j
                storage[i, j] = 2.0
            else
                storage[i, j] = 0.0
            end
        end
    end
end
