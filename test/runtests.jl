using Optim, Compat
using Base.Test

debug_printing = false

general_tests = [
    "api",
    "callables",
    "callbacks",
    "convergence",
    "deprecate",
    "initial_convergence",
    "objective_types",
    "Optim",
    "optimize",
    "type_stability",
    "types",
]
general_tests = map(s->"./general/"*s*".jl", general_tests)

univariate_tests = [
    "optimize/interface",
    "optimize/optimize",
    "solvers/golden_section",
    "solvers/brent",
    #"initial_convergence",
]
univariate_tests = map(s->"./univariate/"*s*".jl", univariate_tests)

multivariate_tests = [
    "solvers/first order/accelerated_gradient_descent",
    "solvers/first order/bfgs",
    "solvers/first order/cg",
    "solvers/first order/gradient_descent",
    "solvers/first order/l_bfgs",
    "solvers/first order/momentum_gradient_descent",
    "solvers/second order/newton",
    "solvers/second order/newton_trust_region",
    "solvers/zeroth order/grid_search",
    "solvers/zeroth order/nelder_mead",
    "solvers/zeroth order/particle_swarm",
    "solvers/zeroth order/simulated_annealing",
    "solvers/constrained/constrained",
    # optimize
    "optimize/interface",
    "optimize/optimize",
    # other
    "array",
    "extrapolate",
    "lsthrow",
    "precon",
]
multivariate_tests = map(s->"./multivariate/"*s*".jl", multivariate_tests)

differentiability_condition(method, prob) = true
differentiability_condition(method::Optim.FirstOrderSolver, prob) = prob.isdifferentiable
differentiability_condition(method::Optim.SecondOrderSolver, prob) = prob.istwicedifferentiable

input_tuple(method, prob) = ((prob.f,),)
input_tuple(method::Optim.FirstOrderSolver, prob) = ((prob.f,), (prob.f, prob.g!))
input_tuple(method::Optim.SecondOrderSolver, prob) = ((prob.f,), (prob.f, prob.g!), (prob.f, prob.g!, prob.h!))

function run_optim_tests(method; convergence_exceptions = (),
                                 minimizer_exceptions = (),
                                 f_increase_exceptions = (),
                                 iteration_exceptions = (),
                                 skip = (),
                                 show_name = false)
    # Loop over unconstrained problems
    for (name, prob) in Optim.UnconstrainedProblems.examples
        show_name && print_with_color(:green, "Problem: ", name, "\n")
        # Look for name in the first elements of the iteration_exceptions tuples
        iter_id = find(n[1] == name for n in iteration_exceptions)
        # If name wasn't found, use default 1000 iterations, else use provided number
        iters = length(iter_id) == 0 ? 1000 : iteration_exceptions[iter_id[1]][2]
        # Construct options
        options = Optim.Options(allow_f_increases = name in f_increase_exceptions, iterations = iters)
        # Check if once or twice differentiable
        if differentiability_condition(method, prob) && !(name in skip)
            # Loop over appropriate input combinations of f, g!, and h!
            for (i, input) in enumerate(input_tuple(method, prob))
                results = Optim.optimize(input..., prob.initial_x, method, options)
                @test isa(summary(results), String)
                if !((name, i) in convergence_exceptions)
                    @test Optim.converged(results)
                end
                if !((name, i) in minimizer_exceptions)
                    @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
                end
            end
        end
    end
end

@testset "general" begin
    for my_test in general_tests
        println(my_test)
        @time include(my_test)
    end
end
@testset "univariate" begin
    for my_test in univariate_tests
        println(my_test)
        @time include(my_test)
    end
end
@testset "multivariate" begin
    for my_test in multivariate_tests
        println(my_test)
        @time include(my_test)
    end
end
