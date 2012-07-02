function gradient_descent2(f::Function,
                           g::Function,
                           initial_x::Vector,
                           tolerance::Float64,
                           alpha::Float64,
                           beta::Float64)
  
  # Set up the initial state of the system.
  x = initial_x
  
  # Count the number of gradient descent steps we perform.
  max_iterations = 1000
  i = 0
  
  # Show trace?
  show_trace = false
  
  if show_trace
    println("Iteration: $(i)")
    println("x: $(x)")
    println("g(x): $(g(x))")
    println("||g(x)||: $(norm(g(x)))")
    println("")
  end
  
  # Monitor convergence.
  converged = false
  
  # Iterate until the norm of the gradient is within tolerance of zero.
  while !converged && i <= max_iterations
    
    # Use a back-tracking line search to select a step-size.
    step_size = backtracking_line_search(f, g, x, -g(x), alpha, beta)
    
    # Move in the direction of the gradient.
    x = x - step_size * g(x)
        
    # Increment the number of steps we've had to perform.
    i = i + 1
    
    if norm(g(x)) <= tolerance
      converged = true
    end
    
    if show_trace
      println("Iteration: $(i)")
      println("x: $(x)")
      println("g(x): $(g(x))")
      println("||g(x)||: $(norm(g(x)))")
      println("Step-size: $(step_size)")
      println("")
    end
  end
  
  # Return the minimum, the function evaluated at the minimum and the
  # number of iterations required to reach the minimum.
  OptimizationResults(initial_x, x, f(x), i, converged)
end
