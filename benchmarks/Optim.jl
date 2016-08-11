f = open(join([version_dir, "optim_benchmark.csv"], "/"), "w")
write(f, join(["Problem", "Optimizer", "Converged", "Time", "Minimum", "Iterations", "f_calls", "g_calls", "f_hat", "f_error", "x_error"], ","))
write(f, "\n")

@showprogress 1 "Benchmarking..." for (name, problem) in Optim.UnconstrainedProblems.examples
    for (i,  algorithm) in enumerate(default_solvers)
        try
        # Force compilation and obtain results
        results = optimize(problem.f, problem.g!, problem.h!,
                           problem.initial_x, algorithm, OptimizationOptions(g_tol = 1e-16))
        # Run each algorithm n times
        n = 10
        if algorithm == ParticleSwarm() && name == "Large Polynomial"
            n = 1
        end
        # Estimate run time in seconds
        run_time = minimum([@elapsed optimize(problem.f, problem.g!, problem.h!,
                               problem.initial_x,
                               algorithm, OptimizationOptions(g_tol = 1e-16)) for nn = 1:n])

        # Count iterations
        iterations = results.iterations

        # Print out results.
        write(f, join([problem.name,
                       default_names[i],
                       Optim.converged(results),
                       run_time,
                       Optim.minimum(results),
                       iterations,
                       Optim.f_calls(results),
                       Optim.g_calls(results),
                       problem.f(problem.solutions),
                       Optim.minimum(results)-problem.f(problem.solutions),
                       norm(Optim.minimizer(results)-problem.solutions, Inf)], ","))
        write(f, "\n")
        catch
            write(f, join([problem.name,
                           default_names[i],
                           "false",
                           "Inf",
                           "Inf",
                           "Inf",
                           "Inf",
                           "Inf",
                           "Inf",
                           "Inf",
                           "Inf"], ","))
            write(f, "\n")
        end
    end
end
close(f)
