# Imports and helpers shared by the test files. runtests.jl includes this first.
# To run a single test file, include it before that file:
#
#     include("test/utils.jl"); include("test/multivariate/solvers/first_order/bfgs.jl")

using Test
using Optim
using OptimTestProblems
using OptimTestProblems.MultivariateProblems
const MVP = MultivariateProblems

import PositiveFactorizations: Positive, cholesky # for the IPNewton tests
using Random

import LineSearches
import NLSolversBase
import NLSolversBase: clear!
import LinearAlgebra: norm, diag, I, Diagonal, dot, eigen, issymmetric, mul!
import SparseArrays: normalize!, spdiagm

import ForwardDiff
import ReverseDiff
using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoReverseDiff

debug_printing = false
test_broken = false

input_tuple(method, prob) = ((MVP.objective(prob),),)
input_tuple(method::Optim.FirstOrderOptimizer, prob) =
    ((MVP.objective(prob),), (MVP.objective(prob), MVP.gradient(prob)))
input_tuple(method::Optim.SecondOrderOptimizer, prob) = (
    (MVP.objective(prob),),
    (MVP.objective(prob), MVP.gradient(prob)),
    (MVP.objective(prob), MVP.gradient(prob), MVP.hessian(prob)),
)
# IPNewton runs the unconstrained problems with infinite bounds, and needs the Hessian
function input_tuple(method::IPNewton, prob)
    prob.istwicedifferentiable || return ()
    df = TwiceDifferentiable(
        MVP.objective(prob),
        MVP.gradient(prob),
        MVP.objective_gradient(prob),
        MVP.hessian(prob),
        prob.initial_x,
    )
    infvec = fill(Inf, size(prob.initial_x))
    return ((df, TwiceDifferentiableConstraints(-infvec, infvec)),)
end

# The first input passes the objective alone, so the gradient is finite
# differenced. Central differences bottom out around eps^(1/3) relative, well
# above the default g_tol of 1e-8, so whether these runs report convergence
# comes down to the rounding of the machine they land on. Accept a gradient
# that is at the finite-difference floor; the minimum and minimizer assertions
# still pin down the answer.
const FD_GRADIENT_TOL = 1e-6

fd_gradient_input(method, i) =
    i == 1 && method isa Union{Optim.FirstOrderOptimizer,Optim.SecondOrderOptimizer}

converged_or_fd_floor(results, method, i) =
    Optim.converged(results) ||
    (fd_gradient_input(method, i) && Optim.g_residual(results) < FD_GRADIENT_TOL)

function test_summary(x, ref::String)
    @test summary(x) == ref
    io = IOBuffer()
    summary(io, x)
    @test String(take!(io)) == ref
    return
end
function test_summary(x)
    summary_x = summary(x)
    @test summary_x isa String
    io = IOBuffer()
    summary(io, x)
    @test String(take!(io)) == summary_x
    return
end

# Rosenbrock from OptimTestProblems with counted objective, gradient and Hessian
# calls, for checking the counts reported in the results.
function counted_rosenbrock()
    prob = MVP.UnconstrainedProblems.examples["Rosenbrock"]
    calls = (f = Ref(0), g = Ref(0), h = Ref(0))
    f(x) = (calls.f[] += 1; MVP.objective(prob)(x))
    g!(out, x) = (calls.g[] += 1; MVP.gradient(prob)(out, x))
    h!(out, x) = (calls.h[] += 1; MVP.hessian(prob)(out, x))
    return prob, f, g!, h!, calls
end

function run_optim_tests(
    method,
    problems = MultivariateProblems.UnconstrainedProblems.examples;
    convergence_exceptions = (),
    minimizer_exceptions = (),
    minimum_exceptions = (),
    iteration_exceptions = (),
    option_overrides = (),
    skip = (),
    show_name = false,
    show_trace = false,
    show_res = false,
    show_itcalls = false,
)
    # Loop over unconstrained problems
    for (name, prob) in problems
        if !isfinite(prob.minimum) || !any(isfinite, prob.solutions)
            debug_printing &&
                println("$name has no registered minimum/minimizer. Skipping ...")
            continue
        end

        show_name && printstyled("Problem: ", name, "\n", color = :green)
        defaults = Optim.default_options(method)
        # Look for name in the first elements of the iteration_exceptions tuples.
        # If it is not there, use the method's default, or else 1000 iterations.
        iter_id = findall(n -> n[1] == name, iteration_exceptions)
        iters =
            length(iter_id) == 0 ? get(defaults, :iterations, 1000) :
            iteration_exceptions[iter_id[1]][2]
        # Look for name in option_overrides: a tuple of (name, NamedTuple) pairs
        # whose contents are splatted into Optim.Options for the matching problem.
        override_id = findall(n -> n[1] == name, option_overrides)
        overrides =
            length(override_id) == 0 ? (;) : option_overrides[override_id[1]][2]
        options = Optim.Options(;
            defaults...,
            iterations = iters,
            show_trace = show_trace,
            overrides...,
        )

        # Use finite difference if it is not differentiable enough
        if !(name in skip)
            for (i, input) in enumerate(input_tuple(method, prob))
                if (!prob.isdifferentiable && i > 1) ||
                   (!prob.istwicedifferentiable && i > 2)
                    continue
                end

                # Loop over appropriate input combinations of f, g!, and h!
                results = Optim.optimize(input..., prob.initial_x, method, options)
                test_summary(results)
                show_res && println(results)
                show_itcalls &&
                    printstyled("Iterations: ", Optim.iterations(results), "\n"; color = :red)
                show_itcalls &&
                    printstyled("f-calls: ", Optim.f_calls(results), "\n"; color = :red)
                show_itcalls &&
                    printstyled("g-calls: ", Optim.g_calls(results), "\n"; color = :red)
                show_itcalls &&
                    printstyled("jvp-calls: ", Optim.jvp_calls(results), "\n"; color = :red)
                show_itcalls &&
                    printstyled("h-calls: ", Optim.h_calls(results), "\n"; color = :red)
                show_itcalls &&
                    printstyled("hvp-calls: ", Optim.hvp_calls(results), "\n"; color = :red)
                if !((name, i) in convergence_exceptions)
                    @test converged_or_fd_floor(results, method, i)
                    # Print on error, easier to debug CI
                    if !converged_or_fd_floor(results, method, i)
                        printstyled(
                            name,
                            " did not converge with i = ",
                            i,
                            "\n",
                            color = :red,
                        )
                        printstyled(results, "\n", color = :red)
                    end
                elseif test_broken
                    @test_broken Optim.converged(results)
                end
                if !((name, i) in minimum_exceptions)
                    @test Optim.minimum(results) <
                          prob.minimum + sqrt(eps(typeof(prob.minimum)))
                elseif test_broken
                    @test_broken Optim.minimum(results) <
                                 prob.minimum + sqrt(eps(typeof(prob.minimum)))
                end
                if !((name, i) in minimizer_exceptions)
                    @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
                elseif test_broken
                    @test_broken norm(Optim.minimizer(results) - prob.solutions) < 1e-2
                end
            end
        else
            @test_broken false    # marked skipped tests as broken
            debug_printing && @info "Skipping $name"
        end
    end
end
