load("src/init.jl")

function f(x)
  (309.0 - 5.0 * x[1])^2 + (17.0 - x[2])^2
end
function g(x)
  [-10.0 * (309.0 - 5.0 * x[1]), -2.0 * (17.0 - x[2])]
end

initial_x = [10.0, 10.0]
tolerance = 10e-8

m = 10

results = l_bfgs(f, g, initial_x, m, tolerance)
@assert norm(results.minimum - [309.0 / 5.0, 17.0]) < 0.01

