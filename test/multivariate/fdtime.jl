@testset "Finite difference timing" begin
    fd_input_tuple(method::Optim.FirstOrderOptimizer, prob) = ((MVP.objective(prob),),)
    fd_input_tuple(method::Optim.SecondOrderOptimizer, prob) = ((MVP.objective(prob),), (MVP.objective(prob), MVP.gradient(prob)))

    function run_optim_fd_tests(method;
                                problems = ("Extended Rosenbrock", "Large Polynomial", "Powell",
                                            "Paraboloid Diagonal", "Penalty Function I",),
                                show_name = false, show_trace = false,
                                show_time = false, show_res = false)

        # Loop over unconstrained problems
        for name in problems
            prob = MVP.UnconstrainedProblems.examples[name]
            show_name && print_with_color(:green, "Problem: ", name, "\n")
            options = Optim.Options(allow_f_increases=true, show_trace = show_trace)
            for (i, input) in enumerate(fd_input_tuple(method, prob))
                # Loop over appropriate input combinations of f, g!, and h!
                results = Optim.optimize(input..., prob.initial_x, method, options)

                debug_printing && print_with_color(:red, "f-calls: $(Optim.f_calls(results))\n")
                show_res && display(results)

                show_time && @time Optim.optimize(input..., prob.initial_x, method, options)

                Optim.converged(results) || @show results
                Optim.converged(results) || @show prob.minimum
                Optim.converged(results) || @show prob.solutions
                @test Optim.converged(results)
                @test Optim.minimum(results) < prob.minimum + sqrt(eps(typeof(prob.minimum)))
                @test norm(Optim.minimizer(results) - prob.solutions) < 1e-2
            end
        end
    end

    @testset "Timing with LBFGS" begin
        debug_printing && print_with_color(:blue, "#####################\nSolver: L-BFGS\n")
        run_optim_fd_tests(LBFGS(), show_name=debug_printing, show_time = debug_printing)
    end

    @testset "Timing with Newton" begin
        debug_printing && print_with_color(:blue, "#####################\nSolver: Newton\n")
        run_optim_fd_tests(Newton(), show_name=debug_printing, show_time = debug_printing)
    end
end
