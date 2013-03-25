##########################################################################
#
# Benchmark optimization algorithms by tracking:
# * Run time over 100 runs -- after 1 initial run that forces JIT.
# * Number of iterations
# * Euclidean error of solution
# * Memory requirements (TODO)
#
##########################################################################

##########################################################################
#
# Load the software and test functions.
#
##########################################################################

using Optim
include("test_functions.jl")

##########################################################################
#
# Iterate over 5 optimizable functions:
# * parabola
# * powell
# * rosenbrock
# * polynomial
# * exponential
#
##########################################################################

problems = Array(Any, 6)

parabola_problem = Dict()
parabola_problem[:name] = "Parabola"
parabola_problem[:f] = parabola
parabola_problem[:g] = parabola_gradient!
parabola_problem[:h] = parabola_hessian!
parabola_problem[:initial_x] = [0.0, 0.0, 0.0, 0.0, 0.0]
parabola_problem[:solution] = [1.0, 2.0, 3.0, 5.0, 8.0]
problems[1] = parabola_problem

powell_problem = Dict()
powell_problem[:name] = "Powell"
powell_problem[:f] = powell
powell_problem[:g] = powell_gradient!
powell_problem[:h] = powell_hessian!
powell_problem[:initial_x] = [3.0, -1.0, 0.0, 1.0]
powell_problem[:solution] = [0.0, 0.0, 0.0, 0.0]
problems[2] = powell_problem

rosenbrock_problem = Dict()
rosenbrock_problem[:name] = "Rosenbrock"
rosenbrock_problem[:f] = rosenbrock
rosenbrock_problem[:g] = rosenbrock_gradient!
rosenbrock_problem[:h] = rosenbrock_hessian!
rosenbrock_problem[:initial_x] = [0.0, 0.0]
rosenbrock_problem[:solution] = [1.0, 1.0]
problems[3] = rosenbrock_problem

polynomial_problem = Dict()
polynomial_problem[:name] = "Polynomial"
polynomial_problem[:f] = polynomial
polynomial_problem[:g] = polynomial_gradient!
polynomial_problem[:h] = polynomial_hessian!
polynomial_problem[:initial_x] = [0.0, 0.0, 0.0]
polynomial_problem[:solution] = [10.0, 7.0, 108.0]
problems[4] = polynomial_problem

exponential_problem = Dict()
exponential_problem[:name] = "Exponential"
exponential_problem[:f] = exponential
exponential_problem[:g] = exponential_gradient!
exponential_problem[:h] = exponential_hessian!
exponential_problem[:initial_x] = [0.0, 0.0]
exponential_problem[:solution] = [2.0, 3.0]
problems[5] = exponential_problem

large_polynomial_problem = Dict()
large_polynomial_problem[:name] = "Large Polynomial"
large_polynomial_problem[:f] = large_polynomial
large_polynomial_problem[:g] = large_polynomial_gradient!
large_polynomial_problem[:h] = large_polynomial_hessian!
large_polynomial_problem[:initial_x] = zeros(250)
large_polynomial_problem[:solution] = [1.0:1.0:250.0]
problems[6] = large_polynomial_problem

##########################################################################
#
# Iterate over 7 optimization functions:
# * naive_gradient_descent: Naive Gradient Descent
# * gradient_descent: Gradient Descent
# * newton: Newton's Method
######## * bfgs: BFGS
# * l_bfgs: L-BFGS
# * nelder_mead: Nelder-Mead
######## * sa: Simulated Annealing
#
##########################################################################

algorithms = [:naive_gradient_descent,
              :gradient_descent,
              :newton,
              #:bfgs,
              :l_bfgs,
              :nelder_mead]
              #:simulated_annealing]

# Print out a header line for the TSV-formatted report.
println(join({"Problem",
              "Algorithm",
              "AverageRunTimeInMilliseconds",
              "Iterations",
              "Error"},
             "\t"))

for problem = problems
  for algorithm = algorithms
    # Force compilation
    results = optimize(problem[:f],
                       problem[:g],
                       problem[:h],
                       problem[:initial_x],
                       algorithm,
                       1e-16,
                       true)
    
    # Run each algorithm 100 times
    n = 100
    
    # Estimate run time in milliseconds
    run_time = @elapsed for i = 1:n
      results = optimize(problem[:f],
                         problem[:g],
                         problem[:h],
                         problem[:initial_x],
                         algorithm,
                         1e-16,
                         true)
    end
    run_time = run_time * n

    # Estimate error in discovered solution
    results = optimize(problem[:f],
                       problem[:g],
                       problem[:h],
                       problem[:initial_x],
                       algorithm,
                       1e-16,
                       true)
    errors = norm(results.minimum - problem[:solution])
    
    # Count iterations
    iterations = results.iterations
    
    # Print out results.
    println(join({problem[:name], results.method, run_time / n, iterations, errors}, "\t"))
  end
end
