srand(1)

function f_s(x::Vector)
    (x[1] - 5.0)^4
end

results = Optim.optimize(f_s, [0.0], method=SimulatedAnnealing(), iterations=100_000)
@assert norm(results.minimum - [5.0]) < 0.1

function rosenbrock_s(x::Vector)
    (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

results = Optim.optimize(rosenbrock_s, [0.0, 0.0], method=SimulatedAnnealing(), iterations=100_000)
@assert norm(results.minimum - [1.0, 1.0]) < 0.1
