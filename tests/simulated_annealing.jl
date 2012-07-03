load("src/init.jl")

srand(1)

function f(x::Vector)
  (x[1] - 5.0)^4
end

results = simulated_annealing(f,
                              [0.0],
                              z -> [rand_uniform(z - 1.0, z + 1.0)],
                              i -> 1 / log(i),
                              true,
                              10e-8,
                              10000,
                              false)
@assert norm(results.minimum - [5.0]) < 0.1

function rosenbrock(x)
  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
 
function neighbors(x)
  [rand_uniform(x[1] - 1, x[1] + 1), rand_uniform(x[2] - 1, x[2] + 1)]
end

results = simulated_annealing(rosenbrock,
                              [0.0, 0.0],
                              neighbors,
                              i -> 1 / log(i),
                              true,
                              10e-8,
                              10000,
                              true)
@assert norm(results.minimum - [1.0, 1.0]) < 0.1

results = simulated_annealing(rosenbrock,
                              [0.0, 0.0],
                              neighbors)
@assert norm(results.minimum - [1.0, 1.0]) < 0.1

results = simulated_annealing(rosenbrock,
                              [0.0, 0.0])
@assert norm(results.minimum - [1.0, 1.0]) < 0.1
