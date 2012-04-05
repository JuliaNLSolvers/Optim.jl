##
#
# gradient_descent2
# * Arguments:
#   * f: A function that is twice-continuously differentiable.
#   * g: The gradient of f.
#   * x0: A value in the domain of f from which to start the search for a minimum.
#   * precision: How close must two successive values of f(x) be for convergence to be declared?
#   * alpha: A number in (0, 0.5) governing the back-tracking line search. (Defaults to 0.1.)
#   * beta: A number in (0, 1) governing the back-gracking line search. (Defaults to 0.8.)
# * Returns:
#   * t: A tuple containing three items:
#     * x_star: The purported minimum of the function to be optimized.
#     * f(x_star): The function evaluated at the returned minimum.
#     * i: The number of iterations required to reach convergence.
#
##

##
#
# Use back-tracking line search to select step size.
#
##

function gradient_descent2(f::Function,
                           g::Function,
                           x0::Any,
                           precision::Float64,
                           alpha::Float64,
                           beta::Float64)
  
  # Set up the initial state of the system.
  # We insure the termination condition is not met by setting y_old = Inf.
  x_old = x0
  x_new = x0
  y_old = Inf
  y_new = f(x_new)
  
  # Count the number of gradient descent steps we perform.
  i = 0
  
  # Iterate until our purported minimum over two passes changes by
  # no more than a prespecified precision.
  while abs(y_new - y_old) > precision
    
    # Use a back-tracking line search to select a step-size.
    step_size = 1
    Dx = -g(x_new)
    g_transpose = g(x_new)'
    
    while f(x_new + step_size * Dx) > f(x_new) + alpha * g_transpose * Dx
      step_size = beta * step_size
    end
    
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
