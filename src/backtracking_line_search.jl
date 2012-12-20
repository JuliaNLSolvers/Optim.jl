using Base

function backtracking_line_search(f::Function,
                                  g::Function,
                                  x::Vector,
                                  p::Vector,
                                  c1::Float64,
                                  c2::Float64,
                                  rho::Float64,
                                  max_iterations::Int64)
  
  # Keep track of the number of iterations.
  i = 0
  
  # Store a copy of the function and gradient evaluted at x.
  f_x = f(x)
  g_x = g(x)
  gxp = dot(g_x,p)
  
  # The default step-size is always 1.
  alpha = 1.0
  
  while (dot(g(x + alpha*p),p) < c2 * gxp) & (alpha < 65536)
    alpha *= 2
  end

  # Keep coming closer to x until we find a point that is as good
  # as the gradient suggests we can do.
  while f(x + alpha*p) > f_x + c1*alpha*gxp
    alpha *= rho
    i += 1    
    if i > max_iterations
      error("Too many iterations in backtracking_line_search.")
    end
  end
  return alpha
end

backtracking_line_search(f::Function,
                         g::Function,
                         x::Vector,
                         p::Vector) = backtracking_line_search(f, g, x, p, 1e-6, 0.9, 0.9, 1000)