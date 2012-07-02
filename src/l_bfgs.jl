function two_loop(x, g_x, rho, s, y, m, k)
  q = g_x
  loop_bounds = 1:min(k, m)
  alpha = zeros(m)
  for i = reverse(loop_bounds)
    alpha[i] = (rho[i] * s[i]' * q)[1]
    q = q - alpha[i] * y[i]
  end
  index = min(k, m)
  gamm = (s[index]' * y[index])[1] / (y[index]' * y[index])[1]
  if isnan(gamm)
    gamm = 1
  end
  r = gamm * q
  for i = loop_bounds
    beta = (rho[i] * y[i]' * r)[1]
    r = r + s[i] * (alpha[i] - beta)
  end
  r
end

# Use Any arrays for the time being
# Eventually move over to a faster Float64 representation for which we built shift and push operations.

function lbfgs(f, g, initial_x, m, tolerance)
  # Set iteration counter.
  k = 1
  
  # Set starting point. We will store x and new_x.
  x = initial_x
  new_x = initial_x

  # Establish size of parameter space.
  n = length(x)
  
  # Initialize rho, s and y.
  rho = zeros(m)
  s = Array(Any, m)
  for i = 1:m
    s[i] = zeros(n)
  end
  #s = zeros(n, m)
  y = Array(Any, m)
  for i = 1:m
    y[i] = zeros(n)
  end
  #y = zeros(n, m)
  
  # Parameters for backtracking line search.
  a = 0.1
  b = 0.8
  
  # Compute the initial gradient.
  g_x = g(x)
  
  # Stop system from going into infinite loop.
  max_iterations = 50
  
  # Print trace information
  show_trace = true
  if show_trace
    println(k)
    println(x)
    println(f(x))
    println()
  end
  
  # Iterate until convergence.
  while norm(g_x) > tolerance && k <= max_iterations
    p = -two_loop(x, g_x, rho, s, y, m, k)
    alpha = backtracking_line_search(f, g, x, p, a, b)
    x_new = x + alpha * p
    tmp_s = x_new - x
    tmp_y = g(x_new) - g(x)
    if k > m # Discard the vector pair s[k - m], y[k - m]
      shift(s)
      push(s, tmp_s)
      shift(y)
      push(y, tmp_y)
      shift(rho)
      push(rho, 1 / (tmp_y' * tmp_s)[1])
    else
      s[k] = tmp_s
      y[k] = tmp_y
      rho[k] = 1 / (tmp_y' * tmp_s)[1]
    end
    x = x_new
    k = k + 1
    if show_trace
      println(k)
      println(x)
      println(f(x))
      println()
    end
  end
end
