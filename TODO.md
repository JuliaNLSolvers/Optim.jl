# General Changes
* Switch over to using `base/distributions.jl` for all RNG's

# Method Changes
* Revise L-BFGS
* Add L-BFGS-B
* Add Brent's method
* Add all methods from Boyd and Vanderberghe's Convex Optimization book
* Add all methods from Nocedal and Wright's book
* Incorporate conjugate gradient code or write from scratch.

# Calculus.jl Changes
* Write function to estimate the gradient of a function numerically
* Write function to estimate the Hessian of a function numerically

# Docs Changes
* Improve documentation

# Testing Changes
* Improve tests using both 1D and 2D problems
* Need to get ridge regression example to produce same solution as `glmnet()` in R
* All methods should be tested and timed on:
  * Rosenbrock
  * Powell's
  * Simple parabola in 5D
* Track run time (after initial compiling run), number of iterations

# API Changes
* All algorithms need to have max iterations set, esp. backtracking line search
* Add new parameters to `gradient_descent()`:
  * `max_iterations`
  * `trace`
* Add new parameters to `gradient_descent2()`:
  * `alpha`, `beta`
  * `max_iterations_gd`
  * `max_iterations_bt`
  * `trace`
* Add new parameters to `newton()`:
  * `alpha`, `beta`
  * `max_iterations_newton`
  * `max_iterations_bt`
  * `trace`
