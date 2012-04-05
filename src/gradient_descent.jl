##
#
# gradient_descent
# * Arguments:
#   * f: A function that is twice-continuously differentiable.
#   * g: The gradient of f.
#   * x0: A value in the domain of f from which to start the search for a minimum.
#   * step_size: How far along the gradient should we move with each step?
#   * precision: How close must two successive values of f(x) be for convergence to be declared?
# * Returns:
#   * t: A tuple containing three items:
#     * x_star: The purported minimum of the function to be optimized.
#     * f(x_star): The function evaluated at the returned minimum.
#     * i: The number of iterations required to reach convergence.
#
##

function gradient_descent(f::Function,
                          g::Function,
                          x0::Any,
                          step_size::Float64,
                          precision::Float64)
  
  # Set up the initial state of the system.
  # We insure the termination condition is not met by setting y_old = Inf.
  x_old = x0
  x_new = x0
  y_old = Inf
  y_new = f(x_new)
  
  i = 0
  
  # Iterate until our purported minimum over two passes changes by
  # no more than a prespecified precision.
  while abs(y_new - y_old) > precision
    x_old = x_new
    x_new = x_new - step_size * g(x_new)
    
    y_old = y_new
    y_new = f(x_new)
    
    i = i + 1
  end
  
  # Return the minimum, the function evaluated at the minimum and the
  # number of iterations required to reach the minimum.
  (x_new, y_new, i)
end
