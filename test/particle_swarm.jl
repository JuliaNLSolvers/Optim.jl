@testset "Particle Swarm" begin
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
    options = Optim.Options(iterations=100)
    res = Optim.optimize(f_s, initial_x, ParticleSwarm(lower, upper, n_particles),
                         options)
    @test norm(Optim.minimizer(res) - [5.0]) < 0.1

    initial_x = [0.0, 0.0]
    lower = [-20., -20.]
    upper = [20., 20.]
    n_particles = 5
    options = Optim.Options(iterations=300)
    res = Optim.optimize(rosenbrock_s, initial_x, ParticleSwarm(lower, upper, n_particles),
                             options)
    @test norm(Optim.minimizer(res) - [1.0, 1.0]) < 0.1

    # Add UnconstrainedProblems here; currently they take too many iterations to be
    # feasible
end
