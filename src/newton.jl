function newton(f, g, h, x0, precision, alpha, beta)
  x = x0
  
  dx = -inv(h(x)) * g(x)
  l2 = g(x)' * inv(h(x)) * g(x)
  
  i = 0
  
  while l2 / 2 > precision
    step_size = backtracking_line_search(f, g, x, dx, alpha, beta)
    
    x = x + step_size * dx
    
    dx = -inv(h(x)) * g(x)
    l2 = g(x)' * inv(h(x)) * g(x)
    
    i = i + 1
  end
  
  (x, f(x), i)
end
