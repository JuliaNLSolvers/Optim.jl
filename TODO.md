* Add L-BFGS-R
* Add all methods from Boyd and Vanderberghe's Convex Optimization book
* Write function to estimate the gradient of a function numerically
* Write function to estimate the Hessian of a function numerically
* Improve documentation
* Write tests; distinguish tests and examples
* Create formal library for easy installation

* Improve tests using both 1D and 2D problems.

* All algorithms need to have max iterations set, esp. backtracking line search.

* Get ridge regression example to produce same solution as glmnet() in R.

* Add options to gradient_descent():
 * max_iterations
 * trace

* Add options to gradient_descent2():
  * alpha, beta
  * max_iterations_gd
  * max_iterations_bt
  * trace

* Add options to newton():
  * alpha, beta
  * max_iterations_newton
  * max_iterations_bt
  * trace
