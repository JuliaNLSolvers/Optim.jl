using Base

function two_loop(x::Vector,
                  g_x::Vector,
                  rho::Vector,
                  s::Vector,
                  y::Vector,
                  m::Int64,
                  k::Int64)
  q = g_x
  
  if k != 1
    loop_bounds = 1:min(k - 1, m)
    alpha = zeros(m)
    for i = reverse(loop_bounds)
      alpha[i] = (rho[i] * s[i]' * q)[1]
      q = q - alpha[i] * y[i]
    end
  end
  
  index = min(k - 1, m)
  if index < 1
    gamm = 1
  else
    gamm = (s[index]' * y[index])[1] / (y[index]' * y[index])[1]
    if isnan(gamm)
      error("NaN-induced freak out!")
    end
  end
  r = gamm * q
  
  if k != 1
    for i = loop_bounds
      beta = (rho[i] * y[i]' * r)[1]
      r = r + s[i] * (alpha[i] - beta)
    end
  end
  
  -r
end

function l_bfgs(f::Function,
                g::Function,
                initial_x::Vector,
                m::Int64,
                tolerance::Float64,
                max_iterations::Int64,
                show_trace::Bool)
  # Set iteration counter.
  k = 1
  
  # Set starting point. We will store x and new_x.
  x = initial_x
  new_x = initial_x
  
  # Establish size of parameter space.
  n = length(x)
  
  # Initialize rho, s and y.
  # Use Any arrays for the time being
  # Eventually move over to a faster Float64 representation
  # for which we build shift and push operations.
  rho = zeros(0)
  s = Array(Any, 0)
  y = Array(Any, 0)
  
  # Compute the initial gradient.
  g_x = g(x)
    
  # Print trace information
  if show_trace
    println("Iteration k: $(k)")
    println("x: $(x)")
    println("f(x): $(f(x))")
    println("g(x): $(g(x))")
    println()
  end
  
  # Iterate until convergence.
  converged = false
  
  while !converged && k < max_iterations
    # Select a search direction.
    p = two_loop(x, g_x, rho, s, y, m, k)
    
    # Select a step-size.
    alpha = backtracking_line_search(f, g, x, p)
    
    # Update position.
    x_new = x + alpha * p
    
    # Estimate movement.
    tmp_s = x_new - x
    tmp_y = g(x_new) - g(x)
    
    # Discard unneeded s, y and rho.
    if k > m
      shift(s)
      shift(y)
      shift(rho)
    end
    
    # Keep a record of the new s, y and rho.
    push(s, tmp_s)
    push(y, tmp_y)
    push(rho, 1 / (tmp_y' * tmp_s)[1])
    
    # Update our position.
    x = x_new
    g_x = g(x)
    
    # Update the iteration counter.
    k = k + 1
    
    # Assess convergence.
    if norm(g_x) <= tolerance
      converged = true
    end
    
    # Show trace.
    if show_trace
      println("Iteration: $(k)")
      println("x: $(x)")
      println("f(x): $(f(x))")
      println("g(x): $(g(x))")
      println()
    end
  end
  
  OptimizationResults("L-BFGS", initial_x, x, f(x), k, converged)
end

function l_bfgs(f::Function, g::Function, initial_x::Vector)
  l_bfgs(f, g, initial_x, 10, 10e-8, 1000, false)
end
