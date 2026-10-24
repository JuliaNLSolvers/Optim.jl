include("utils.jl")

special_tests = ["bigfloat/initial_convergence"]
special_tests = map(s -> "./special/" * s * ".jl", special_tests)

general_tests = [
    "api",
    "callables",
    "callbacks",
    "convergence",
    "default_solvers",
    "deprecate",
    "initial_convergence",
    "objective_types",
    "optimize",
    "type_stability",
    "types",
    "counter",
    "maximize",
]
general_tests = map(s -> "./general/" * s * ".jl", general_tests)

univariate_tests = [
    # optimize
    "optimize/interface",
    "optimize/optimize",
    # solvers
    "solvers/golden_section",
    "solvers/brent",
    # "initial_convergence",
    "dual",
]
univariate_tests = map(s -> "./univariate/" * s * ".jl", univariate_tests)

multivariate_tests = [
    "lsthrow",
    ## optimize
    "optimize/interface",
    "optimize/optimize",
    "optimize/inplace",
    ## solvers
    ## constrained
    "solvers/constrained/fminbox",
    "solvers/constrained/lbfgsb",
    "solvers/constrained/ipnewton/interface",
    "solvers/constrained/ipnewton/constraints",
    "solvers/constrained/ipnewton/counter",
    "solvers/constrained/ipnewton/ipnewton_unconstrained",
    "solvers/constrained/samin",
    ## first order
    "solvers/first_order/accelerated_gradient_descent",
    "solvers/first_order/adam_adamax",
    "solvers/first_order/bfgs",
    "solvers/first_order/cg",
    "solvers/first_order/gradient_descent",
    "solvers/first_order/l_bfgs",
    "solvers/first_order/momentum_gradient_descent",
    "solvers/first_order/ngmres",
    ## second order
    "solvers/second_order/newton",
    "solvers/second_order/newton_trust_region",
    "solvers/second_order/krylov_trust_region",
    ## zeroth order
    "solvers/zeroth_order/grid_search",
    "solvers/zeroth_order/nelder_mead",
    "solvers/zeroth_order/particle_swarm",
    "solvers/zeroth_order/simulated_annealing",
    ## other
    "array",
    "extrapolate",
    "linesearch_handoff",
    "precon",
    "manifolds",
    "complex",
    "arbitrary_precision",
    "successive_f_tol",
    "f_increase",
    "measurements",
]
multivariate_tests = map(s -> "./multivariate/" * s * ".jl", multivariate_tests)

@testset verbose = true "Optim.jl" begin
    @testset "special" begin
        @testset for my_test in special_tests
            println(my_test)
            @time include(my_test)
        end
    end
    @testset "general" begin
        @testset for my_test in general_tests
            println(my_test)
            @time include(my_test)
        end
    end
    @testset "univariate" begin
        @testset for my_test in univariate_tests
            println(my_test)
            @time include(my_test)
        end
    end
    @testset "multivariate" begin
        @testset for my_test in multivariate_tests
            println(my_test)
            @time include(my_test)
        end
    end

    println("Literate examples")
    @time include("examples.jl")

    @testset "show method for options" begin
        o = Optim.Options()
        @test occursin(" = ", sprint(show, o))
    end

    @testset "MOI wrapper" begin
        include("MOI_wrapper.jl")
    end

    @testset "QA" begin
        include("qa.jl")
    end
end
