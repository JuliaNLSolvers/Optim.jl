function update_hessian(h::Matrix, s::Vector, y::Vector)
  rho = 1.0 / (y' * s)[1]
  I = eye(size(h, 1))
  (I - rho * s * y') * h * (I - rho * y * s') + rho * s * s'
end

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
  if show_trace
    println("Iteration k: $(k)")
    println("x_new: $(x_new)")
    println("f(x_new): $(f(x_new))")
    println("g(x_new): $(g(x_new))")
    println("h: $(h)")
    println()
  end
  
  while !converged && k < max_iterations
    # Increment the iteration counter.
    k = k + 1
    
    # Set the search direction.
    p = -h * gradient_new
    
    # Calculate a step-size.
    alpha = backtracking_line_search(f, g, x_new, p)
    
    # Update our position.
    x_old = x_new
    x_new = x_old + alpha * p
    s = x_new - x_old
    
    # Update the gradient.
    gradient_old = gradient_new
    gradient_new = g(x_new)
    y = gradient_new - gradient_old
    
    # Update the Hessian.
    h = update_hessian(h, s, y)
    
    # Assess convergence.
    if norm(gradient_new) <= tolerance
      converged = true
    end
    
    # Show state of the system.
    if show_trace
      println("Iteration k: $(k)")
      println("x_new: $(x_new)")
      println("f(x_new): $(f(x_new))")
      println("g(x_new): $(g(x_new))")
      println("h: $(h)")
      println()
    end
  end
  
  OptimizationResults("BFGS", initial_x, x_new, f(x_new), k, converged)
end

function bfgs(f::Function,
              g::Function,
              initial_x::Vector,
              initial_h::Matrix)
  bfgs(f, g, initial_x, initial_h, 10e-8, 1000, false)
end

function bfgs(f::Function,
              g::Function,
              initial_x::Vector)
  n = length(initial_x)
  bfgs(f, g, initial_x, eye(n), 10e-8, 1000, false)
end
