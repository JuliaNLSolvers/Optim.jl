# naive_gradient_descent()

* Arguments:
  * f: A function that is differentiable.
  * g: The gradient of f.
  * x0: A value in the domain of f from which to start the search for a minimum.
  * step_size: How far along the gradient should we move with each step?
  * tolerance: How close must two successive values of f(x) be for convergence to be declared?
* Returns:
  * A tuple containing three items:
    * x_star: The purported minimum of the function to be optimized.
    * f(x_star): The function evaluated at the returned minimum.
    * i: The number of iterations required to reach convergence.
