function newton(f::Function,
                g::Function,
                h::Function,
                x0::Vector,
                tolerance::Float64,
                alpha::Float64,
                beta::Float64)
  
  x = x0
  
  dx = -inv(h(x)) * g(x)
  l2 = (g(x)' * inv(h(x)) * g(x))[1]
  
  # Don't run forever.
  max_iterations = 1000  
  i = 0
  
  while norm(g(x)) > tolerance && i <= max_iterations
    dx = -inv(h(x)) * g(x)
    
    step_size = backtracking_line_search(f, g, x, dx, alpha, beta)
    
    x = x + step_size * dx
    
    l2 = (g(x)' * inv(h(x)) * g(x))[1]
    
    i = i + 1
  end
  
  (x, f(x), i)
end
