# newton()

* Arguments:
  * f: A function that is twice-differentiable.
  * g: The gradient of f.
  * h: The Hessian of f.
  * x0: A value in the domain of f from which to start the search for a minimum.
  * tolerance: How close must two successive values of f(x) be for convergence to be declared?
  * alpha: Parameter of the back-tracking line search method.
  * beta: Parameter of the back-tracking line search method.
* Returns:
  * A tuple containing three items:
    * x_star: The purported minimum of the function to be optimized.
    * f(x_star): The function evaluated at the returned minimum.
    * i: The number of iterations required to reach convergence.
