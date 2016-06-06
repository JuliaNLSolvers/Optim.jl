##########################################################################
#
# Benchmark optimization algorithms by tracking:
#
# * Run time over 100 runs -- after 1 initial run that forces JIT.
# * Number of iterations
# * Euclidean error of solution
# * Memory requirements (TODO)
#
##########################################################################

using Optim

algorithms = [:gradient_descent,
              :momentum_gradient_descent,
              :bfgs,
              :l_bfgs,
              :cg]

# Print out a header line for the TSV-formatted report
println(join({"Problem",
              "Algorithm",
              "AverageRunTimeInMilliseconds",
              "Iterations",
              "Error",
              "Autodiff"},
              "\t"))

for (name, problem) in Optim.UnconstrainedProblems.examples
    for algorithm in algorithms
        for ad in [false, true]
        if !problem.isdifferentiable &&
              (algorithm == :gradient_descent ||
               algorithm == :momentum_gradient_descent ||
               algorithm == :bfgs ||
               algorithm == :l_bfgs ||
               algorithm == :cg)
            continue
        end

        # Force compilation
        results = optimize(problem.f,
                           problem.initial_x,
                           method = algorithm,
                           g_tol = 1e-8,
                           autodiff = ad)

        # Run each algorithm 1,000 times
        n = 1_000

        # Estimate run time in seconds
        run_time = @elapsed for i = 1:n
            results = optimize(problem.f,
                               problem.initial_x,
                               method = algorithm,
                               g_tol = 1e-8,
                               autodiff = ad)
        end

        # Estimate error in discovered solution
        results = optimize(problem.f,
                           problem.initial_x,
                           method = algorithm,
                           g_tol = 1e-8,
                           autodiff = ad)
        errors = minimum(map(sol -> norm(results.minimum - sol), problem.solutions))

        # Count iterations
        iterations = results.iterations

        # Print out results.
        println(join({problem.name,
                      results.method,
                      run_time,
                      iterations,
                      errors,
                      ad},
                      "\t"))
        end
    end
end
