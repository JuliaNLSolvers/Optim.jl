# nelder_mead()

* Arguments:
  * f: A function.
  * initial_p: n + 1 initial points defining a simplex in N-dimensional space.
  * a: Reflection parameter.
  * g: Expansion parameter.
  * b: Contraction parameter.
  * tolerance: How small must SD of simplex be to declare convergence?
  * max_iterations: How many iterations before giving up?
  * trace: Print out a trace of each step?
* Returns:
  * A tuple containing three items:
    * x_star: The purported minimum of the function to be optimized.
    * f(x_star): The function evaluated at the returned minimum.
    * i: The number of iterations required to reach convergence.
