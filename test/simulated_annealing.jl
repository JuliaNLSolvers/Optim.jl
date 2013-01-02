load("Optim")
using Optim

srand(1)

function f(x::Vector)
    (x[1] - 5.0)^4
end

results = Optim.simulated_annealing(f,
                                    [0.0],
                                    z -> [Optim.rand_uniform(z - 1.0, z + 1.0)],
                                    i -> 1 / log(i),
                                    true,
                                    10e-8,
                                    10000,
                                    true,
                                    false)
@assert norm(results.minimum - [5.0]) < 0.1

function rosenbrock(x)
    (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function neighbors(x)
    [Optim.rand_uniform(x[1] - 1, x[1] + 1), Optim.rand_uniform(x[2] - 1, x[2] + 1)]
end

results = Optim.simulated_annealing(rosenbrock,
                                    [0.0, 0.0],
                                    neighbors,
                                    i -> 1 / log(i),
                                    true,
                                    10e-8,
                                    10000,
                                    true,
                                    false)
@assert norm(results.minimum - [1.0, 1.0]) < 0.1

results = Optim.simulated_annealing(rosenbrock,
                                    [0.0, 0.0],
                                    neighbors)
@assert norm(results.minimum - [1.0, 1.0]) < 0.1

results = Optim.simulated_annealing(rosenbrock,
                                    [0.0, 0.0])
@assert norm(results.minimum - [1.0, 1.0]) < 0.1
