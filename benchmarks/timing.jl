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
              :newton,
              :bfgs,
              :l_bfgs,
              :cg,
              :nelder_mead,
              :simulated_annealing]

# Print out a header line for the TSV-formatted report
println(join(["Problem",
              "Algorithm",
              "AverageRunTimeInMilliseconds",
              "Iterations",
              "Error"],
              "\t"))

for (name, problem) in Optim.UnconstrainedProblems.examples
    for algorithm in algorithms
        if !problem.istwicedifferentiable && algorithm == :newton
            continue
        end
        if !problem.isdifferentiable &&
              (algorithm == :gradient_descent ||
               algorithm == :momentum_gradient_descent ||
               algorithm == :l_bfgs ||
               algorithm == :bfgs ||
               algorithm == :cg ||
               algorithm == :newton)
            continue
        end
        if problem.name == "Large Polynomial" && algorithm == :nelder_mead
            continue
        end

        # Force compilation
        results = optimize(problem.f,
                           problem.g!,
                           problem.h!,
                           problem.initial_x,
                           method = algorithm,
                           grtol = 1e-16)

        # Run each algorithm 1,000 times
        n = 1_000

        # Estimate run time in seconds
        run_time = @elapsed for i = 1:n
            results = optimize(problem.f,
                               problem.g!,
                               problem.h!,
                               problem.initial_x,
                               method = algorithm,
                               grtol = 1e-16)
        end

        # Estimate error in discovered solution
        results = optimize(problem.f,
                           problem.g!,
                           problem.h!,
                           problem.initial_x,
                           method = algorithm,
                           grtol = 1e-16)
        errors = minimum(map(sol -> norm(results.minimum - sol), problem.solutions))

        # Count iterations
        iterations = results.iterations

        # Print out results.
        println(join([problem.name,
                      results.method,
                      run_time,
                      iterations,
                      errors],
                      "\t"))
    end
end
