using Base

function gradient_descent(f::Function,
                          g::Function,
                          initial_x::Vector,
                          tolerance::Float64,
                          max_iterations::Int64,
                          show_trace::Bool)
  
  # Set up the initial state of the system.
  x = initial_x
  
  # Count the number of gradient descent steps we perform.
  i = 0
  
  # Show trace.
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
  while !converged && i < max_iterations
    # Increment the number of steps we've had to perform.
    i = i + 1
    
    # Use a back-tracking line search to select a step-size.
    step_size = backtracking_line_search(f, g, x, -g(x))
    
    # Move in the direction of the gradient.
    x = x - step_size * g(x)
    
    # Assess convergence.
    if norm(g(x)) <= tolerance
      converged = true
    end
    
    # Show trace.
    if show_trace
      println("Iteration: $(i)")
      println("x: $(x)")
      println("g(x): $(g(x))")
      println("||g(x)||: $(norm(g(x)))")
      println("Step-size: $(step_size)")
      println("")
    end
  end
  
  OptimizationResults("Gradient Descent w/ Backtracking Line Search", initial_x, x, f(x), i, converged)
end

function gradient_descent(f::Function,
                          g::Function,
                          initial_x::Vector)
  gradient_descent(f, g, initial_x, 10e-8, 1000, false)
end
