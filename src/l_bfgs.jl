# Use Any arrays for the time being
# Eventually move over to a faster Float64 representation for which we built shift and push operations.

# while k < m, search all the way back from 1:k
# while k >= m, search back for m copies starting k and going back to k - m + 1

# In first pass, just use gradient.

function two_loop(x, g_x, rho, s, y, m, k)
  q = g_x
    
  if k == 1
    1 # NO-OP
  else
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
  
  if k == 1
    1 # NO-OP
  else
    for i = loop_bounds
      beta = (rho[i] * y[i]' * r)[1]
      r = r + s[i] * (alpha[i] - beta)
    end
  end
  
  r
end

function l_bfgs(f, g, initial_x, m, tolerance)
  # Set iteration counter.
  k = 1
  
  # Set starting point. We will store x and new_x.
  x = initial_x
  new_x = initial_x

  # Establish size of parameter space.
  n = length(x)
  
  # Initialize rho, s and y.
  rho = zeros(0)
  s = Array(Any, 0)
  #for i = 1:m
  #  s[i] = zeros(n)
  #end
  #s = zeros(n, m)
  y = Array(Any, 0)
  #for i = 1:m
  #  y[i] = zeros(n)
  #end
  #y = zeros(n, m)
  
  # Parameters for backtracking line search.
  a = 0.1
  b = 0.8
  
  # Compute the initial gradient.
  g_x = g(x)
  
  # Stop system from going into infinite loop.
  max_iterations = 10
  
  # Print trace information
  show_trace = false
  if show_trace
    println("Iteration k: $(k)")
    println("x: $(x)")
    println("f(x): $(f(x))")
    println("g(x): $(g(x))")
    println()
  end
  
  # Iterate until convergence.
  converged = false
  
  while !converged && k <= max_iterations
    p = -two_loop(x, g_x, rho, s, y, m, k)
    
    alpha = backtracking_line_search(f, g, x, p, a, b)
    
    x_new = x + alpha * p
    
    tmp_s = x_new - x
    tmp_y = g(x_new) - g(x)
    
    # Discard the vector pair s[k - m], y[k - m]
    if k > m
      shift(s)
      shift(y)
      shift(rho)
    end
    
    push(s, tmp_s)
    push(y, tmp_y)
    push(rho, 1 / (tmp_y' * tmp_s)[1])

    x = x_new
    g_x = g(x)
    k = k + 1
    
    if show_trace
      println("Iteration k: $(k)")
      println("x: $(x)")
      println("f(x): $(f(x))")
      println("g(x): $(g(x))")
      println()
    end
    
    if norm(g_x) <= tolerance
      converged = true
    end
  end
  
  OptimizationResults(initial_x, x, f(x), k, converged)
end

function l_bfgs(f, g, initial_x)
  l_bfgs(f, g, initial_x, 10, 10e-8)
end
