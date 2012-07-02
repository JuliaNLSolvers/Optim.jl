# Iterate over optimizable functions:
# * fletcher_powell
# * parabola
# * powell
# * rosenbrock

# Iterate over optimization functions:
# * gradient_descent
# * gradient_descent2
# * newton
# * bfgs
# * l_bfgs
# * nelder_mead
# * simulated_annealing

# Use same initial point for all algorithms. Change across problems.

# Track:
# * Run time over 100 runs -- after initial compiling run
# * Number of iterations
# * Euclidean error of solution
# * Memory requirements

load("src/init.jl")
load("testbed/test_functions.jl")

println(join({"Problem", "Algorithm", "AverageRunTimeInMilliseconds", "Iterations", "Error"}, "\t"))

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

println(join({"Parabola", "Gradient Descent", run_time / n, iterations, errors}, "\t"))

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

println(join({"Parabola", "Gradient Descent 2", run_time / n, iterations, errors}, "\t"))

###
### Newton's Method
###

# Force compilation
results = newton(parabola, parabola_gradient, parabola_hessian, zeros(5), 10e-8, 0.1, 0.8)

# Estimate run time
run_time = @elapsed for i = 1:n
  results = newton(parabola, parabola_gradient, parabola_hessian, zeros(5), 10e-8, 0.1, 0.8)
end
run_time = run_time * 1000

# Estimate error
results = newton(parabola, parabola_gradient, parabola_hessian, zeros(5), 10e-8, 0.1, 0.8)
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "Newton's Method", run_time / n, iterations, errors}, "\t"))

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

println(join({"Parabola", "BFGS", run_time / n, iterations, errors}, "\t"))

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

println(join({"Parabola", "L-BFGS", run_time / n, iterations, errors}, "\t"))

###
### Nelder-Mead
###

exit()

a = 1.0
g = 2.0
b = 0.5

# Force compilation
initial_points = randn(6, 5)
results = nelder_mead(parabola, initial_points, a, g, b, 10e-8, 1000, false)

# Estimate run time
run_time = @elapsed for i = 1:n
  results = l_bfgs(parabola, parabola_gradient, zeros(5), 10, 10e-8)
end
run_time = run_time * 1000

# Estimate error
results = l_bfgs(parabola, parabola_gradient, zeros(5), 10, 10e-8)
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "L-BFGS", run_time / n, iterations, errors}, "\t"))

###
### Simulated Annealing
###

exit()

# Force compilation
function neighbors(x)
  [rand_uniform(x[1] - 1, x[1] + 1),
   rand_uniform(x[2] - 1, x[2] + 1),
   rand_uniform(x[3] - 1, x[3] + 1),
   rand_uniform(x[4] - 1, x[4] + 1),
   rand_uniform(x[5] - 1, x[5] + 1)]
end

results = simulated_annealing(parabola,
                              zeros(5),
                              neighbors,
                              i -> 1 / log(i),
                              100,
                              true,
                              false)

# Estimate run time
run_time = @elapsed for i = 1:n
  results = l_bfgs(parabola, parabola_gradient, zeros(5), 10, 10e-8)
end
run_time = run_time * 1000

# Estimate error
results = l_bfgs(parabola, parabola_gradient, zeros(5), 10, 10e-8)
errors = norm(results.minimum - [1.0, 2.0, 3.0, 5.0, 8.0])

iterations = results.iterations

println(join({"Parabola", "Simulated Annealing", run_time / n, iterations, errors}, "\t"))
