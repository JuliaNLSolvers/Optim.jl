load("src/optim.jl")
load("/Users/johnmyleswhite/julia_libs/rng.jl")

srand(1)

function rosenbrock(x, y)
  (1 - x)^2 + 100(y - x^2)^2
end
 
function neighbors(z)
  [rand_uniform(z[1] - 1, z[1] + 1), rand_uniform(z[2] - 1, z[2] + 1)]
end
 
solution = simulated_annealing(z -> rosenbrock(z[1], z[2]),
                               [0, 0],
                               neighbors,
                               i -> 1 / log(i),
                               10000,
                               true,
                               false)

@assert abs(solution[1] - 1) < 0.1
@assert abs(solution[2] - 1) < 0.1
