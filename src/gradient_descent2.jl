function gradient_descent2(f::Function,
                           g::Function,
                           x0::Any,
                           tolerance::Float64,
                           alpha::Float64,
                           beta::Float64)
  
  # Set up the initial state of the system.
  x_old = x0
  x_new = x0
  
  # Count the number of gradient descent steps we perform.
  i = 0
  max_iterations = 1000
  
  # Iterate until the norm of the gradient is within tolerance of zero.
  while any(sqrt(g(x_new)' * g(x_new)) > tolerance) && i <= max_iterations
    
    # Use a back-tracking line search to select a step-size.
    step_size = backtracking_line_search(f, g, x_new, -g(x_new), alpha, beta)
    
    # Move in the direction of the gradient.
    x_old = x_new
    x_new = x_new - step_size * g(x_new)
    
	# if trace
	#println(x_old)
	#println(x_new)
	#println(step_size)
	#println("")
	
    # Increment the number of steps we've had to perform.
    i = i + 1
  end
  
  # Return the minimum, the function evaluated at the minimum and the
  # number of iterations required to reach the minimum.
  (x_new, f(x_new), i)
end
