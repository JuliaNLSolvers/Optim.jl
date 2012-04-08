function newton(f, g, h, x0, tolerance, alpha, beta)
  x = x0
  
  dx = -inv(h(x)) * g(x)
  l2 = g(x)' * inv(h(x)) * g(x)
  
  i = 0
  max_iterations = 1000
  
  while any(l2 / 2 > tolerance) && i <= max_iterations
    step_size = backtracking_line_search(f, g, x, dx, alpha, beta)
    
    x = x + step_size * dx
    
    dx = -inv(h(x)) * g(x)
    l2 = g(x)' * inv(h(x)) * g(x)
    
    i = i + 1
  end
  
  (x, f(x), i)
end
