function backtracking_line_search(f, g, x, dx, alpha, beta) 
  i = 0
  max_iterations = 1000
  
  t = 1
  
  while any(f(x + t * dx) > f(x) + alpha * g(x)' * dx)
    t = beta * t
    
    i = i + 1
    if i > max_iterations
      error("Too many iterations in backtracking_line_search(alpha: $alpha, beta: $beta)")
    end
  end
  
  t
end
