using Optim, Compat
if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end
debug_printing = false

my_tests = [
    "types.jl",
    "bfgs.jl",
    "gradient_descent.jl",
    "accelerated_gradient_descent.jl",
    "momentum_gradient_descent.jl",
    "grid_search.jl",
    "l_bfgs.jl",
    "newton.jl",
    "newton_trust_region.jl",
    "cg.jl",
    "nelder_mead.jl",
    "optimize.jl",
    "simulated_annealing.jl",
    "particle_swarm.jl",
    "golden_section.jl",
    "brent.jl",
    "type_stability.jl",
    "array.jl",
    "constrained.jl",
    "callbacks.jl",
    "precon.jl",
    "initial_convergence.jl",
    "extrapolate.jl",
    "lsthrow.jl",
    "api.jl",
]

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

for my_test in my_tests
    @time include(my_test)
end
