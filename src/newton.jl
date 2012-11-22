using Base

function newton(f::Function,
                g::Function,
                h::Function,
                initial_x::Vector,
                tolerance::Float64,
                max_iterations::Int64,
                show_trace::Bool)
  
  # Maintain a record of the state.
  x = initial_x
  
  # Select a stepsize.
  dx = -inv(h(x)) * g(x)
  l2 = (g(x)' * inv(h(x)) * g(x))[1]
  
  # Don't run forever.
  i = 0
  
  # Track convergence.
  converged = false
  
  # Show state of the system.
  if show_trace
    println("Iteration: $(i)")
    println("x: $(x)")
    println("f(x): $(f(x))")
    println("g(x): $(g(x))")
    println("h(x): $(h(x))")
    println()
  end
  
  while !converged && i < max_iterations
    # Update the iteration counter.
    i = i + 1
    
    # Select a search direction.
    dx = -inv(h(x)) * g(x)
    
    # Select a step size.
    step_size = backtracking_line_search(f, g, x, dx)
    
    # Update our position.
    x = x + step_size * dx
    
    # Assess converged convergence.
    l2 = (g(x)' * inv(h(x)) * g(x))[1]
    if l2 / 2 <= tolerance
      converged = true
    end
    
    # Show state of the system.
    if show_trace
      println("Iteration: $(i)")
      println("x: $(x)")
      println("f(x): $(f(x))")
      println("g(x): $(g(x))")
      println("h(x): $(h(x))")
      println()
    end
  end
  
  OptimizationResults("Newton's Method", initial_x, x, f(x), i, converged)
end

function newton(f::Function,
                g::Function,
                h::Function,
                initial_x::Vector)
  newton(f, g, h, initial_x, 10e-16, 1000, false)
end
