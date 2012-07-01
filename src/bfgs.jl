function update_h(h, s, y)
  rho = 1 / (y' * s)[1]
  I = eye(size(h, 1))
  (I - rho * s * y') * h * (I - rho * y * s') + rho * s * s'
end

function bfgs(f, g, initial_x, initial_h, tolerance)
  k = 0
  
  x_new = initial_x
  x_old = initial_x
  
  gradient_new = g(x_new)
  gradient_old = g(x_old)
  
  h = initial_h
  
  max_iterations = 1000
  
  a = 0.1
  b = 0.8
  
  while norm(gradient_new) > tolerance && k <= max_iterations
    p = -h * gradient_new
    alpha = backtracking_line_search(f, g, x_new, p, a, b)
    x_old = x_new
    x_new = x_old + alpha * p
    s = x_new - x_old
    gradient_old = gradient_new
    gradient_new = g(x_new)
    y = gradient_new - gradient_old
    h = update_h(h, s, y)
    k = k + 1
    println(k)
    println(x_new)
    println(f(x_new))
    println()
  end
end
