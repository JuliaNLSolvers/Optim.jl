load("src/init.jl")

function f(x)
  x[1]^2 + (2 - x[2])^2
end
function g(x)
  [2x[1], -(2 - x[2])]
end

initial_x = [10.0, 10.0]
tolerance = 10e-8

m = 10

lbfgs(f, g, initial_x, m, tolerance)
