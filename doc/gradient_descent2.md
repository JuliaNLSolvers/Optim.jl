# gradient_descent2()

* Arguments:
  * f: A function that is twice-continuously differentiable.
  * g: The gradient of f.
  * x0: A value in the domain of f from which to start the search for a minimum.
  * precision: How close must two successive values of f(x) be for convergence to be declared?
  * alpha: A number in (0, 0.5) governing the back-tracking line search. (Defaults to 0.1.)
  * beta: A number in (0, 1) governing the back-gracking line search. (Defaults to 0.8.)
* Returns:
  * t: A tuple containing three items:
    * x_star: The purported minimum of the function to be optimized.
    * f(x_star): The function evaluated at the returned minimum.
    * i: The number of iterations required to reach convergence.
