module UnivariateProblems

    ### Sources
    ###
    ### [1]  http://infinity77.net/global_optimization/test_functions_1d.html

    struct UnivariateProblem
        name::AbstractString
        f::Function
        bounds::Vector{Float64}
        minimizers::Vector{Float64}
        minima::Vector{Float64}
    end

    examples = Dict{AbstractString, UnivariateProblem}()

    f(x) = 2x^2+3x+1

    examples["Polynomial"] = UnivariateProblem("Polynomial",
                                                  f,
                                                  [-2.0, 1.0],
                                                  [-0.75,],
                                                  [f(-0.75),])

    # Problem 04 from [1]
    p04(x) = -(16x^2-24x+5)exp(-x)

    examples["Problem04"] = UnivariateProblem("Problem04",
                                                p04,
                                                [1.9, 3.9],
                                                [2.868034,],
                                                [p04(2.868034),])

    # Problem 13 from [1]
    p13(x) = -x^(2/3)-(1-x^2)^(1/3)

    examples["Problem13"] = UnivariateProblem("Problem13",
                                                p13,
                                                [0.001, 0.99],
                                                [1./sqrt(2.0),],
                                                [p13(1./sqrt(2.0)),])

    # Problem 18 from [1]
    p18(x) = x <= 3.0 ? (x-2.0)^2 : 2.0*log(x-2.0)+1.0

    examples["Problem18"] = UnivariateProblem("Problem18",
                                                p18,
                                                [0.0, 6.0],
                                                [2.0,],
                                                [p18(2.0),])
end
