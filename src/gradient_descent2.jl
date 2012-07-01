function gradient_descent2(f::Function,
                           g::Function,
                           x0::Vector,
                           tolerance::Float64,
                           alpha::Float64,
                           beta::Float64)
  
  # Set up the initial state of the system.
  x = x0
  
  # Count the number of gradient descent steps we perform.
  max_iterations = 1000
  i = 0
  
  # Show trace?
  show_trace = false
  
  # Iterate until the norm of the gradient is within tolerance of zero.
  while norm(g(x)) > tolerance && i <= max_iterations
    
    # Use a back-tracking line search to select a step-size.
    step_size = backtracking_line_search(f, g, x, -g(x), alpha, beta)
    
    # Move in the direction of the gradient.
    x = x - step_size * g(x)
    
	  if show_trace
      println(i)
      println(x)
      println(step_size)
      println("")
    end
    
    # Increment the number of steps we've had to perform.
    i = i + 1
  end
  
  # Return the minimum, the function evaluated at the minimum and the
  # number of iterations required to reach the minimum.
  (x, f(x), i)
end
