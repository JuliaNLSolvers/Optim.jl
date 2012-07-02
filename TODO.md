* Switch over to using `base/distributions.jl` for all RNG's
* Revise L-BFGS
* Add L-BFGS-B
* Add all methods from Boyd and Vanderberghe's Convex Optimization book
* Add all methods from Nocedl and Wright's book
* Write function to estimate the gradient of a function numerically
* Write function to estimate the Hessian of a function numerically
* Improve documentation
* Improve tests using both 1D and 2D problems
* All algorithms need to have max iterations set, esp. backtracking line search
* Need to get ridge regression example to produce same solution as `glmnet()` in R
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
* Incorporate conjugate gradient code or write from scratch.

* Return object of type OptimizationResults.
