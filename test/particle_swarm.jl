srand(100)

function f_s(x::Vector)
    (x[1] - 5.0)^4
end

function rosenbrock_s(x::Vector)
    (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end


initial_x = [0.0]
upper = [100.0]
lower = [-100.0]
n_particles = 4

res = Optim.optimize(f_s, initial_x, method=ParticleSwarm(lower, upper, n_particles),
                     iterations=100)
@assert norm(Optim.minimizer(res) - [5.0]) < 0.1

initial_x = [0.0, 0.0]
lower = [-20., -20.]
upper = [20., 20.]
n_particles = 5
res = Optim.optimize(rosenbrock_s, initial_x, method=ParticleSwarm(lower, upper, n_particles),
                         iterations=300)
@assert norm(Optim.minimizer(res) - [1.0, 1.0]) < 0.1
