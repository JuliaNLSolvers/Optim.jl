function gradient_descent(f::Function,
                          g::Function,
                          initial_x::Vector,
                          step_size::Float64,
                          tolerance::Float64)
  
  # Set up the initial state of the system.
  # We insure the termination condition is not met by setting y_old = Inf.
  x_old = initial_x
  x_new = initial_x
  y_old = Inf
  y_new = f(x_new)
  
  # Don't go for more than 1,000 iterations.
  max_iterations = 1000
  i = 0
  
  # Track convergence.
  converged = false
  
  # Iterate until our purported minimum over two passes changes by
  # no more than a prespecified tolerance.
  while !converged && i <= max_iterations
    x_old = x_new
    x_new = x_new - step_size * g(x_new)
    
    y_old = y_new
    y_new = f(x_new)
    
    i = i + 1
    
    if abs(y_new - y_old) <= tolerance
      converged = true
    end
  end
  
  # Return the minimum, the function evaluated at the minimum and the
  # number of iterations required to reach the minimum.
  OptimizationResults(initial_x, x_new, y_new, i, converged)
end
