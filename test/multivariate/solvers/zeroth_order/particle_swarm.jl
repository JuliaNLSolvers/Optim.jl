@testset "Particle Swarm" begin
    # TODO: Run on MultivariateProblems.UnconstrainedProblems?
    Random.seed!(100)

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
    @suppress_out begin
        options = Optim.Options(iterations=300, show_trace=true, extended_trace=true, store_trace=true)
        res = Optim.optimize(rosenbrock_s, initial_x, ParticleSwarm(lower, upper, n_particles), options)
        @test summary(res) == "Particle Swarm"
        res = Optim.optimize(rosenbrock_s, initial_x, ParticleSwarm(n_particles = n_particles), options)
        @test summary(res) == "Particle Swarm"
    end
end
