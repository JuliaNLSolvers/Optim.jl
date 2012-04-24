function gradient_descent(f::Function,
                          g::Function,
                          x0::Any,
                          step_size::Float64,
                          tolerance::Float64)
  
  # Set up the initial state of the system.
  # We insure the termination condition is not met by setting y_old = Inf.
  x_old = x0
  x_new = x0
  y_old = Inf
  y_new = f(x_new)
  
  i = 0
  max_iterations = 1000
  
  # Iterate until our purported minimum over two passes changes by
  # no more than a prespecified tolerance.
  while abs(y_new - y_old) > tolerance && i <= max_iterations
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
