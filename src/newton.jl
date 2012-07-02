function newton(f::Function,
                g::Function,
                h::Function,
                initial_x::Vector,
                tolerance::Float64,
                alpha::Float64,
                beta::Float64)
  
  x = initial_x
  
  dx = -inv(h(x)) * g(x)
  l2 = (g(x)' * inv(h(x)) * g(x))[1]
  
  # Don't run forever.
  max_iterations = 1000  
  i = 0
  
  # Track convergence.
  converged = false
  
  while !converged && i <= max_iterations
    dx = -inv(h(x)) * g(x)
    
    step_size = backtracking_line_search(f, g, x, dx, alpha, beta)
    
    x = x + step_size * dx
    
    l2 = (g(x)' * inv(h(x)) * g(x))[1]
    
    i = i + 1
    
    if l2 / 2 <= tolerance
      converged = true
    end
  end
  
  OptimizationResults(initial_x, x, f(x), i, converged)
end

function newton(f::Function,
                g::Function,
                h::Function,
                initial_x::Vector)
  newton(f, g, h, initial_x, 10e-8, 0.1, 0.8)
end
