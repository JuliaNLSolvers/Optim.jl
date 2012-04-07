function backtracking_line_search(f, g, x, dx, alpha, beta)
  t = 1
  while any(f(x + t * dx) > f(x) + alpha * g(x)' * dx)
    t = beta * t
  end
  t
end
