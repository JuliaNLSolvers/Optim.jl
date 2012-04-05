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
  x_old = x0
  x_new = x0
  
  # Count the number of gradient descent steps we perform.
  i = 0
  
  # Iterate until the norm of the gradient is within precision of zero.
  while sqrt(g(x_new)' * g(x_new)) > precision
    
    # Use a back-tracking line search to select a step-size.
    step_size = backtracking_line_search(f, g, x_new, -g(x_new), alpha, beta)
    
    # Move in the direction of the gradient.
    x_old = x_new
    x_new = x_new - step_size * g(x_new)
    
    # Increment the number of steps we've had to perform.
    i = i + 1
  end
  
  # Return the minimum, the function evaluated at the minimum and the
  # number of iterations required to reach the minimum.
  (x_new, f(x_new), i)
end
