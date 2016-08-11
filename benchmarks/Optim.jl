# Print out a header line for the TSV-formatted report
f = open(join([pkgdir, "benchmarks", version_sha, "optim_benchmark.csv"], "/"), "w")
write(f, join(["Problem", "Optimizer", "Converged", "Time", "Minimum", "Iterations", "f_calls", "g_calls"], ","))
write(f, "\n")

    for (i,  algorithm) in enumerate(default_solvers)
        println(algorithm)
        @showprogress 1 "Benchmarking..." for (name, problem) in Optim.UnconstrainedProblems.examples
        try

        # Force compilation
        results = optimize(problem.f,
                           problem.g!,
                           problem.h!,
                           problem.initial_x,
                           method = algorithm,
                           g_tol = 1e-16)
        # Run each algorithm 1,000 times
        n = 10
        if algorithm == ParticleSwarm() && name == "Large Polynomial"
            n = 1
        end
        # Estimate run time in seconds
        run_time = minimum([@elapsed optimize(problem.f,
                               problem.g!,
                               problem.h!,
                               problem.initial_x;
                               method = algorithm,
                               g_tol = 1e-16) for nn = 1:n])

        # Estimate error in discovered solution
        results = optimize(problem.f,
                           problem.g!,
                           problem.h!,
                           problem.initial_x,
                           method = algorithm,
                           g_tol = 1e-16)
#        errors = minimum(map(sol -> norm(results.minimum - sol), problem.solutions))

        # Count iterations
        iterations = results.iterations

        # Print out results.
        write(f, join([problem.name,
                       default_names[i], # TODO fixme to use custom names
                       Optim.converged(results),
                       run_time,
                       Optim.minimum(results),
                       iterations,
                       Optim.f_calls(results),
                       Optim.g_calls(results),
                       0.0], ","))
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
                           0.0], ","))
            write(f, "\n")
        end
    end
end
close(f)
