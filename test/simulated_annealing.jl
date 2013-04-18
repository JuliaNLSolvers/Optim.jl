srand(1)

function f(x::Vector)
    (x[1] - 5.0)^4
end

results = Optim.simulated_annealing(f, [0.0])
@assert norm(results.minimum - [5.0]) < 0.1

function rosenbrock(x::Vector)
    (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

results = Optim.simulated_annealing(rosenbrock, [0.0, 0.0])
@assert norm(results.minimum - [1.0, 1.0]) < 0.1
