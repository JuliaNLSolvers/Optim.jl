function bfgs(f::Function,
              g::Function,
              initial_x::Vector,
              initial_h::Matrix,
              tolerance::Float64,
              max_iterations::Int64,
              show_trace::Bool)
  
  # Keep track of the number of iterations.
  k = 0
  
  # Keep a record of our current position.
  x_new = initial_x
  x_old = initial_x
  
  # Keep a record of the current gradient.
  gradient_new = g(x_new)
  gradient_old = g(x_old)
  
  # Initialize our approximate Hessian.
  h = initial_h
  
  # Iterate until convergence.
  converged = false
  
  # Show state of the system.
  # if show_trace
  #   println("Iteration k: $(k)")
  #   println("x_new: $(x_new)")
  #   println("f(x_new): $(f(x_new))")
  #   println("g(x_new): $(g(x_new))")
  #   println("h: $(h)")
  #   println()
  # end
  
  while !converged && k < max_iterations
    # Increment the iteration counter.
    k += 1
    
    # Set the search direction.
    p = -h * gradient_new
    
    # Calculate a step-size.
    alpha = backtracking_line_search(f, g, x_new, p)
    
    # Show state of the system.
    if show_trace
      # println("Iteration k: $(k)")
      # println("x_new: $(x_new)")
      # println("f(x_new): $(f(x_new))")
      # println("g(x_new): $(g(x_new))")
      # println("h: $(h)")
      # println()
      @printf("Iteration: %-6d f(x): %10.3e\tStep-size: %8.5f\tFirst-order opt.:%10.3e\n", k, f(x_old), alpha, norm(gradient_old, Inf))
    end

    # Update our position.
    x_old = x_new
    x_new = x_old + alpha * p
    s = x_new - x_old

    # Update the gradient.
    gradient_old = gradient_new
    gradient_new = g(x_new)
    y = gradient_new - gradient_old
    
    # Update the Hessian.
    rho = 1.0 / dot(y,s)
    if rho == Inf
      println("Cannot decrease the objective function along the current search direction")
      break
    end
    v = eye(size(h, 1)) - rho * y * s'
    h = (k == 1 ? v'v * dot(y,s)/dot(y,y) : v' * h * v + rho * s * s')
    
    # Assess convergence.
    if norm(gradient_new, Inf) <= tolerance
      converged = true
    end
  end
  
  OptimizationResults("BFGS", initial_x, x_new, f(x_new), k, converged)
end

bfgs(f::Function,
     g::Function,
     initial_x::Vector,
     initial_h::Matrix,
     show_trace::Bool) = bfgs(f, g, initial_x, initial_h, 10e-8, 1000, show_trace)

bfgs(f::Function,
     g::Function,
     initial_x::Vector,
     initial_h::Matrix) = bfgs(f, g, initial_x, initial_h, false)

bfgs(f::Function,
     g::Function,
     initial_x::Vector,
     show_trace::Bool) = bfgs(f, g, initial_x, eye(length(initial_x)), show_trace)

bfgs(f::Function,
     g::Function,
     initial_x::Vector) = bfgs(f, g, initial_x, false)
