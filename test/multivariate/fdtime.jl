@testset "Finite difference timing" begin
    fd_input_tuple(method::Optim.FirstOrderOptimizer, prob) = ((UP.objective(prob),),)
    fd_input_tuple(method::Optim.SecondOrderOptimizer, prob) = ((UP.objective(prob),), (UP.objective(prob), UP.gradient(prob)))

    function run_optim_fd_tests(method; convergence_exceptions = (),
                                minimizer_exceptions = (),
                                minimum_exceptions = (),
                                f_increase_exceptions = (),
                                iteration_exceptions = (),
                                skip = (),
                                show_name = false,
                                show_trace = false,
                                show_res = false)
        # Loop over unconstrained problems
        for (name, prob) in OptimTestProblems.UnconstrainedProblems.examples
            if !isfinite(prob.minimum) || !any(isfinite, prob.solutions)
                debug_printing && println("$name has no registered minimum/minimizer. Skipping ...")
                continue
            end
            show_name && print_with_color(:green, "Problem: ", name, "\n")
            # Look for name in the first elements of the iteration_exceptions tuples
            iter_id = find(n[1] == name for n in iteration_exceptions)
            # If name wasn't found, use default 1000 iterations, else use provided number
            iters = length(iter_id) == 0 ? 1000 : iteration_exceptions[iter_id[1]][2]
            # Construct options
            options = Optim.Options(allow_f_increases = name in f_increase_exceptions, iterations = iters, show_trace = show_trace)

            # Use finite difference if it is not differentiable enough
            if  !(name in skip)
                for (i, input) in enumerate(fd_input_tuple(method, prob))
                    if (!prob.isdifferentiable && i > 1) || (!prob.istwicedifferentiable && i > 2)
                        continue
                    end

                    # Loop over appropriate input combinations of f, g!, and h!
                    results = Optim.optimize(input..., prob.initial_x, method, options)
                    debug_printing && print_with_color(:red, "f-calls: $(Optim.f_calls(results))\n")
                    @time Optim.optimize(input..., prob.initial_x, method, options)

                    show_res && println(results)
                    if !((name, i) in convergence_exceptions)
                        @test Optim.converged(results)
                    end
                    if !((name, i) in minimum_exceptions)
                        @test Optim.minimum(results) < prob.minimum + sqrt(eps(typeof(prob.minimum)))
                    end
                    if !((name, i) in minimizer_exceptions)
                        @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
                    end
                end
            else
                debug_printing && print_with_color(:blue, "Skipping $name\n")
            end
        end
    end

    skip = ("Trigonometric",)
    @testset "Timing with LBFGS" begin
        debug_printing && print_with_color(:blue, "#####################\nSolver: L-BFGS\n")
        run_optim_fd_tests(LBFGS(), skip = skip,
                           show_name=debug_printing)
    end

    @testset "Timing with Newton" begin
        debug_printing && print_with_color(:blue, "#####################\nSolver: Newton\n")
        run_optim_fd_tests(Newton(), skip = skip,
                           show_name=debug_printing)
    end
end
