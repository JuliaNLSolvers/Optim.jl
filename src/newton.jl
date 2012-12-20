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

  # Don't run forever.
  i = 0

  # Show state of the system.
  if show_trace
    println("Iteration: $(i)")
    println("x: $(x)")
    println("f(x): $(f(x))")
    println("g(x): $(g(x))")
    println("h(x): $(h(x))")
    println()
  end
  
  # Track convergence.
  converged = false

  # Select a stepsize.
  dx = -h(x)\g(x)
  l2 = dot(g(x), -dx)

  while !converged && i < max_iterations
    # Update the iteration counter.
    i += 1

    # Select a step size.
    step_size = backtracking_line_search(f, g, x, dx)

    # Update our position.
    x += step_size * dx
    
    # Show state of the system.
    if show_trace
      println("Iteration: $(i)")
      println("x: $(x)")
      println("f(x): $(f(x))")
      println("g(x): $(g(x))")
      println("h(x): $(h(x))")
      println()
    end

    # Select a search direction.
    dx = -h(x)\g(x)

    # Assess convergence.
    l2 = dot(g(x),-dx)
    if l2 / 2 <= tolerance
      converged = true
    end
  end
  
  OptimizationResults("Newton's Method", initial_x, x, f(x), i, converged)
end

newton(f::Function,
       g::Function,
       h::Function,
       initial_x::Vector,
       show_trace::Bool) = newton(f, g, h, initial_x, 10e-16, 1000, show_trace)
newton(f::Function,
       g::Function,
       h::Function,
       initial_x::Vector) = newton(f, g, h, initial_x, false)

