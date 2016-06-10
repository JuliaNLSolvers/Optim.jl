srand(100)

function f_s(x::Vector)
    (x[1] - 5.0)^4
end

function rosenbrock_s(x::Vector)
    (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end


x0 = [0.0]
xmax = [100.0]
xmin = [-100.0]
n_particles = 4

res = Optim.optimize(f_s, x0, method=ParticleSwarm(xmin, xmax, n_particles),
                     iterations=100)
@assert norm(Optim.minimizer(res) - [5.0]) < 0.1

x0 = [0.0, 0.0]
xmin = [-20., -20.]
xmax = [20., 20.]
n_particles = 5
res = Optim.optimize(rosenbrock_s, x0, method=ParticleSwarm(xmin, xmax, n_particles),
                         iterations=300)
@assert norm(Optim.minimizer(res) - [1.0, 1.0]) < 0.1
