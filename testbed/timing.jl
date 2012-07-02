# Iterate over optimization functions:
# * gradient_descent
# * gradient_descent2
# * newton
# * bfgs
# * l_bfgs
# * nelder_mead
# * simulated_annealing

# Iterate over optimizable functions:
# * fletcher_powell
# * parabola
# * powell
# * rosenbrock

# Use same initial point for all algorithms. Change across problems.

# Track:
# * Run time over 100 runs -- after initial compiling run
# * Number of iterations
# * Euclidean error of solution
# * Memory requirements

load("src/init.jl")
load("testbed/test_functions.jl")

println(join({"Algorithm", "Average Run Time in Milliseconds", "Iterations", "Error"}, "\t"))

# Go for 10,000 runs of each algorithm.
n = 10_000

###
### Gradient Descent
###

# Force compilation
results = gradient_descent(parabola, parabola_gradient, zeros(5), 0.1, 10e-8)

# Estimate run time
run_time = @elapsed for i = 1:n
  results = gradient_descent(parabola, parabola_gradient, zeros(5), 0.1, 10e-8)
end
run_time = run_time * 1000

# Estimate error
results = gradient_descent(parabola, parabola_gradient, zeros(5), 0.1, 10e-8)
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Gradient Descent", run_time / n, iterations, errors}, "\t"))

###
### Gradient Descent 2
###

# Force compilation
results = gradient_descent2(parabola, parabola_gradient, zeros(5), 10e-8, 0.1, 0.8)

# Estimate run time
run_time = @elapsed for i = 1:n
  results = gradient_descent2(parabola, parabola_gradient, zeros(5), 10e-8, 0.1, 0.8)
end
run_time = run_time * 1000

# Estimate error
results = gradient_descent2(parabola, parabola_gradient, zeros(5), 10e-8, 0.1, 0.8)
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Gradient Descent 2", run_time / n, iterations, errors}, "\t"))

###
### Newton's Method
###

###
### BFGS
###

# Force compilation
results = bfgs(parabola, parabola_gradient, zeros(5), eye(5), 10e-8)

# Estimate run time
run_time = @elapsed for i = 1:n
  results = bfgs(parabola, parabola_gradient, zeros(5), eye(5), 10e-8)
end
run_time = run_time * 1000

# Estimate error
results = bfgs(parabola, parabola_gradient, zeros(5), eye(5), 10e-8)
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"BFGS", run_time / n, iterations, errors}, "\t"))

###
### L-BFGS
###

# Force compilation
results = l_bfgs(parabola, parabola_gradient, zeros(5), 10, 10e-8)

# Estimate run time
run_time = @elapsed for i = 1:n
  results = l_bfgs(parabola, parabola_gradient, zeros(5), 10, 10e-8)
end
run_time = run_time * 1000

# Estimate error
results = l_bfgs(parabola, parabola_gradient, zeros(5), 10, 10e-8)
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"L-BFGS", run_time / n, iterations, errors}, "\t"))

###
### Nelder-Mead
###

###
### Simulated Annealing
###
