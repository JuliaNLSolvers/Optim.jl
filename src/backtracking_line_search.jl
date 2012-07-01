function backtracking_line_search(f::Function,
                                  g::Function,
                                  x::Vector,
                                  dx::Vector,
                                  alpha::Float64,
                                  beta::Float64)
  
  # Keep track of the number of iterations.
  max_iterations = 1000
  i = 0
  
  # The default upper bound is always 1.
  t = 1
  
  # Store a copy of the function and gradient evaluted at x.
  f_x = f(x)
  g_x = g(x)
  angle = (g_x' * dx)[1]
  
  # Keep coming closer until we find a point that is as good as the gradient suggests.
  while f(x + t * dx) > f_x + alpha * t * angle
    t = beta * t
    
    i = i + 1
    
    if i > max_iterations
      error("Too many iterations in backtracking_line_search(alpha: $alpha, beta: $beta)")
    end
  end
  
  t
end
