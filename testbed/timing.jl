##########################################################################
#
# Iterate over optimizable functions:
# * fletcher_powell
# * parabola
# * powell
# * rosenbrock
#
##########################################################################

##########################################################################
#
# Iterate over optimization functions:
# * gradient_descent
# * gradient_descent
# * newton
# * bfgs
# * l_bfgs
# * nelder_mead
# * simulated_annealing
#
##########################################################################

##########################################################################
#
# Use same initial point for all algorithms. Change across problems.
#
##########################################################################

##########################################################################
#
# Track:
# * Run time over 100 runs -- after initial compiling run
# * Number of iterations
# * Euclidean error of solution
# * Memory requirements
#
##########################################################################

load("src/init.jl")
load("testbed/test_functions.jl")

println(join({"Problem", "Algorithm", "AverageRunTimeInMilliseconds", "Iterations", "Error"}, "\t"))

# Go for 10,000 runs of each algorithm.
n = 10_000

##########################################################################
###
### Constant Stepsize Gradient Descent
###
##########################################################################

# Force compilation
results = naive_gradient_descent(parabola, parabola_gradient, zeros(5))

# Estimate run time
run_time = @elapsed for i = 1:n
  results = naive_gradient_descent(parabola, parabola_gradient, zeros(5))
end
run_time = run_time * 1000

# Estimate error
results = naive_gradient_descent(parabola, parabola_gradient, zeros(5))
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "Naive Gradient Descent", run_time / n, iterations, errors}, "\t"))

##########################################################################
###
### Gradient Descent 2
###
##########################################################################

# Force compilation
results = gradient_descent(parabola, parabola_gradient, zeros(5))

# Estimate run time
run_time = @elapsed for i = 1:n
  results = gradient_descent(parabola, parabola_gradient, zeros(5))
end
run_time = run_time * 1000

# Estimate error
results = gradient_descent(parabola, parabola_gradient, zeros(5))
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "Gradient Descent", run_time / n, iterations, errors}, "\t"))

##########################################################################
###
### Newton's Method
###
##########################################################################

# Force compilation
results = newton(parabola, parabola_gradient, parabola_hessian, zeros(5))

# Estimate run time
run_time = @elapsed for i = 1:n
  results = newton(parabola, parabola_gradient, parabola_hessian, zeros(5))
end
run_time = run_time * 1000

# Estimate error
results = newton(parabola, parabola_gradient, parabola_hessian, zeros(5))
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "Newton's Method", run_time / n, iterations, errors}, "\t"))

##########################################################################
###
### BFGS
###
##########################################################################

# Force compilation
results = bfgs(parabola, parabola_gradient, zeros(5))

# Estimate run time
run_time = @elapsed for i = 1:n
  results = bfgs(parabola, parabola_gradient, zeros(5))
end
run_time = run_time * 1000

# Estimate error
results = bfgs(parabola, parabola_gradient, zeros(5))
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "BFGS", run_time / n, iterations, errors}, "\t"))

##########################################################################
###
### L-BFGS
###
##########################################################################

# Force compilation
results = l_bfgs(parabola, parabola_gradient, zeros(5))

# Estimate run time
run_time = @elapsed for i = 1:n
  results = l_bfgs(parabola, parabola_gradient, zeros(5))
end
run_time = run_time * 1000

# Estimate error
results = l_bfgs(parabola, parabola_gradient, zeros(5))
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "L-BFGS", run_time / n, iterations, errors}, "\t"))

##########################################################################
###
### Nelder-Mead
###
##########################################################################

# Force compilation
initial_points = randn(6, 5)
results = nelder_mead(parabola, initial_points)

# Estimate run time
run_time = @elapsed for i = 1:n
  results = nelder_mead(parabola, initial_points)
end
run_time = run_time * 1000

# Estimate error
results = nelder_mead(parabola, initial_points)
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "Nelder-Mead", run_time / n, iterations, errors}, "\t"))

##########################################################################
###
### Simulated Annealing
###
##########################################################################

# Force compilation
results = simulated_annealing(parabola, zeros(5))

# Estimate run time. Only use 100 runs for SA because of its slowness.
run_time = @elapsed for i = 1:100
  results = simulated_annealing(parabola, zeros(5))
end
run_time = run_time * 1000

# Estimate error
results = simulated_annealing(parabola, zeros(5))
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "Simulated Annealing", run_time / 100, iterations, errors}, "\t"))

##########################################################################
###
### Newton's method on other problems.
###
##########################################################################

#results = newton(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, zeros(2))
#results = newton(powell, powell_gradient, powell_hessian, zeros(5))
#results = newton(fletcher_powell, fletcher_powell_gradient, fletcher_powell_hessian, zeros(5))
