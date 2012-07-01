load("src/init.jl")

function f(x)
  x[1]^2 + (2 - x[2])^2
end
function g(x)
  [2x[1], -(2 - x[2])]
end

initial_x = [100.0, 100.0]
initial_h = eye(2)
tolerance = 10e-8

bfgs(f, g, initial_x, initial_h, tolerance)
