function naive_gradient_descent(f::Function,
                                g::Function,
                                initial_x::Vector,
                                step_size::Float64,
                                tolerance::Float64,
                                max_iterations::Int64,
                                show_trace::Bool)
  
  # Set up the initial state of the system.
  # We insure the termination condition is not met by setting y_old = Inf.
  x_old = initial_x
  x_new = initial_x
  y_old = Inf
  y_new = f(x_new)
  
  # Keep track of the number of iterations.
  i = 0
  
  # Track convergence.
  converged = false
  
  # Show trace.
  if show_trace
    println("Iteration: $(i)")
    println("x_new: $(x_new)")
    println("f(x_new): $(f(x_new))")
    println("g(x_new): $(g(x_new))")
    println("")
  end
  
  # Iterate until our purported minimum over two passes changes by
  # no more than a prespecified tolerance.
  while !converged && i < max_iterations
    # Update the iteration counter.
    i = i + 1
    
    # Update our position.
    x_old = x_new
    x_new = x_new - step_size * g(x_new)
    
    # Update the cached function value.
    y_old = y_new
    y_new = f(x_new)
    
    # Assess convergence.
    if abs(y_new - y_old) <= tolerance
      converged = true
    end
    
    # Show trace.
    if show_trace
      println("Iteration: $(i)")
      println("x_new: $(x_new)")
      println("f(x_new): $(f(x_new))")
      println("g(x_new): $(g(x_new))")
      println("")
    end
  end
  
  OptimizationResults(initial_x, x_new, y_new, i, converged)
end

# Set default tolerance, max_iterations and show_trace.
function naive_gradient_descent(f::Function,
                                g::Function,
                                initial_x::Vector,
                                step_size::Float64)
  naive_gradient_descent(f, g, initial_x, step_size, 10e-8, 1000, false)
end

# Set default step_size, tolerance, max_iterations and show_trace.
function naive_gradient_descent(f::Function,
                                g::Function,
                                initial_x::Vector)
  naive_gradient_descent(f, g, initial_x, 0.1, 10e-8, 1000, false)
end
