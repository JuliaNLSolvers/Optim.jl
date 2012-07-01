load("src/init.jl")

function f(x)
  (1.0 - x[1])^2
end

function g(x)
  [-2.0 * (1.0 - x[1])]
end

x = [0.0]
dx = -g(x)

alpha = 0.1
beta = 0.8

t = backtracking_line_search(f, g, x, dx, alpha, beta)

@assert f(x + t * dx) < f(x) + alpha * t * (g(x)' * dx)[1]
