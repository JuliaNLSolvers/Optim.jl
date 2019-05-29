using Test
using Optim
using OptimTestProblems
using OptimTestProblems.MultivariateProblems
const MVP = MultivariateProblems

using Suppressor
import PositiveFactorizations: Positive, cholesky # for the IPNewton tests
using Random

import LineSearches
import ForwardDiff
import NLSolversBase
import NLSolversBase: clear!
import LinearAlgebra: norm, diag, I, Diagonal, dot, eigen, issymmetric, mul!
import SparseArrays: normalize!, spdiagm

debug_printing = false

special_tests = [
    "bigfloat/initial_convergence",
]
special_tests = map(s->"./special/"*s*".jl", special_tests)

general_tests = [
    "api",
    "callables",
    "callbacks",
    "convergence",
    "default_solvers",
    "deprecate",
    "initial_convergence",
    "objective_types",
    "Optim",
    "optimize",
    "type_stability",
#    "types",
    "counter",
    "maximize",
]
general_tests = map(s->"./general/"*s*".jl", general_tests)

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
univariate_tests = map(s->"./univariate/"*s*".jl", univariate_tests)

multivariate_tests = [
    ## optimize
    "optimize/interface",
    "optimize/optimize",
    "optimize/inplace",
    ## solvers
    ## constrained
    "solvers/constrained/fminbox",
    "solvers/constrained/ipnewton/constraints",
    "solvers/constrained/ipnewton/counter",
    "solvers/constrained/ipnewton/ipnewton_unconstrained",
    "solvers/constrained/samin",
    ## first order
    "solvers/first_order/accelerated_gradient_descent",
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
    "lsthrow",
    "precon",
    "manifolds",
    "complex",
    "fdtime",
    "arbitrary_precision",
]
multivariate_tests = map(s->"./multivariate/"*s*".jl", multivariate_tests)

input_tuple(method, prob) = ((MVP.objective(prob),),)
input_tuple(method::Optim.FirstOrderOptimizer, prob) = ((MVP.objective(prob),), (MVP.objective(prob), MVP.gradient(prob)))
input_tuple(method::Optim.SecondOrderOptimizer, prob) = ((MVP.objective(prob),), (MVP.objective(prob), MVP.gradient(prob)), (MVP.objective(prob), MVP.gradient(prob), MVP.hessian(prob)))

function run_optim_tests(method; convergence_exceptions = (),
                         minimizer_exceptions = (),
                         minimum_exceptions = (),
                         f_increase_exceptions = (),
                         iteration_exceptions = (),
                         skip = (),
                         show_name = false,
                         show_trace = false,
                         show_res = false,
                         show_itcalls = false)
    # Loop over unconstrained problems
    for (name, prob) in MultivariateProblems.UnconstrainedProblems.examples
        if !isfinite(prob.minimum) || !any(isfinite, prob.solutions)
            debug_printing && println("$name has no registered minimum/minimizer. Skipping ...")
            continue
        end

        show_name && printstyled("Problem: ", name, "\n", color=:green)
        # Look for name in the first elements of the iteration_exceptions tuples
        iter_id = findall(n->n[1] == name, iteration_exceptions)
        # If name wasn't found, use default 1000 iterations, else use provided number
        iters = length(iter_id) == 0 ? 1000 : iteration_exceptions[iter_id[1]][2]
        # Construct options
        allow_f_increases = (name in f_increase_exceptions)
        dopts = Optim.default_options(method)
        if haskey(dopts, :allow_f_increases)
            allow_f_increases = allow_f_increases || dopts[:allow_f_increases]
            delete!(dopts, :allow_f_increases)
        end
        options = Optim.Options(allow_f_increases = allow_f_increases,
                                iterations = iters, show_trace = show_trace;
                                dopts...)

        # Use finite difference if it is not differentiable enough
        if  !(name in skip)
            for (i, input) in enumerate(input_tuple(method, prob))
                if (!prob.isdifferentiable && i > 1) || (!prob.istwicedifferentiable && i > 2)
                    continue
                end

                # Loop over appropriate input combinations of f, g!, and h!
                results = Optim.optimize(input..., prob.initial_x, method, options)
                @test isa(summary(results), String)
                show_res && println(results)
                show_itcalls && printstyled("Iterations: $(Optim.iterations(results))\n", color=:red)
                show_itcalls && printstyled("f-calls: $(Optim.f_calls(results))\n", color=:red)
                show_itcalls && printstyled("g-calls: $(Optim.g_calls(results))\n", color=:red)
                show_itcalls && printstyled("h-calls: $(Optim.h_calls(results))\n", color=:red)
                if !((name, i) in convergence_exceptions)
                    @test Optim.converged(results)
                    # Print on error, easier to debug CI
                    if !(Optim.converged(results))
                        printstyled(name, " did not converge with i = ", i, "\n", color=:red)
                        printstyled(results, "\n", color=:red)
                    end
                end
                if !((name, i) in minimum_exceptions)
                    @test Optim.minimum(results) < prob.minimum + sqrt(eps(typeof(prob.minimum)))
                end
                if !((name, i) in minimizer_exceptions)
                    @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
                end
            end
        else
            debug_printing && printstyled("Skipping $name\n", color=:blue)
        end
    end
end

function run_optim_tests_constrained(method; convergence_exceptions = (),
                         minimizer_exceptions = (),
                         minimum_exceptions = (),
                         f_increase_exceptions = (),
                         iteration_exceptions = (),
                         skip = (),
                         show_name = false,
                         show_trace = false,
                         show_res = false,
                         show_itcalls = false)
    # TODO: Update with constraint problems too?
    # Loop over unconstrained problems
    for (name, prob) in MVP.UnconstrainedProblems.examples
        if !isfinite(prob.minimum) || !any(isfinite, prob.solutions)
            debug_printing && println("$name has no registered minimum/minimizer. Skipping ...")
            continue
        end
        show_name && printstyled("Problem: ", name, "\n", color=:green)
        # Look for name in the first elements of the iteration_exceptions tuples
        iter_id = findall(n->n[1] == name, iteration_exceptions)
        # If name wasn't found, use default 1000 iterations, else use provided number
        iters = length(iter_id) == 0 ? 1000 : iteration_exceptions[iter_id[1]][2]
        # Construct options
        allow_f_increases = (name in f_increase_exceptions)
        options = Optim.Options(iterations = iters, show_trace = show_trace; Optim.default_options(method)...)

        # Use finite difference if it is not differentiable enough
        if  !(name in skip) && prob.istwicedifferentiable
            # Loop over appropriate input combinations of f, g!, and h!
            df = TwiceDifferentiable(MVP.objective(prob), MVP.gradient(prob),
                                     MVP.objective_gradient(prob), MVP.hessian(prob), prob.initial_x)
            infvec = fill(Inf, size(prob.initial_x))
            constraints = TwiceDifferentiableConstraints(-infvec, infvec)
            results = optimize(df,constraints,prob.initial_x, method, options)
            @test isa(Optim.summary(results), String)
            show_res && println(results)
            show_itcalls && printstyled("Iterations: $(Optim.iterations(results))\n", color=:red)
            show_itcalls && printstyled("f-calls: $(Optim.f_calls(results))\n", color=:red)
            show_itcalls && printstyled("g-calls: $(Optim.g_calls(results))\n", color=:red)
            show_itcalls && printstyled("h-calls: $(Optim.h_calls(results))\n", color=:red)
            if !(name in convergence_exceptions)
                @test Optim.converged(results)
                # Print on error
                if !(Optim.converged(results))
                    printstyled(name, "did not converge\n", color=:red)
                    printstyled(results, "\n", color=:red)
                end
            end
            if !(name in minimum_exceptions)
                @test Optim.minimum(results) < prob.minimum + sqrt(eps(typeof(prob.minimum)))
            end
            if !(name in minimizer_exceptions)
                @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
            end
        else
            debug_printing && printstyled("Skipping $name\n", color=:blue)
        end
    end
end


@testset "special" begin
    for my_test in special_tests
        println(my_test)
        @time include(my_test)
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

println("Literate examples")
@time include("examples.jl")

@testset "show method for options" begin
    o = Optim.Options()
    @test occursin(" = ", sprint(show, o))
end
