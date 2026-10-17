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
    options = Optim.Options(iterations = 100)
    res = Optim.optimize(f_s, initial_x, ParticleSwarm(lower, upper, n_particles), options)
    @test norm(Optim.minimizer(res) - [5.0]) < 0.1

    initial_x = [0.0, 0.0]
    lower = [-20.0, -20.0]
    upper = [20.0, 20.0]
    n_particles = 5
    options = Optim.Options(iterations = 300)
    res = Optim.optimize(
        rosenbrock_s,
        initial_x,
        ParticleSwarm(lower, upper, n_particles),
        options,
    )
    @test norm(Optim.minimizer(res) - [1.0, 1.0]) < 0.1
    options = Optim.Options(
        iterations = 10,
        show_trace = true,
        extended_trace = true,
        store_trace = true,
    )
    res = Optim.optimize(
        rosenbrock_s,
        initial_x,
        ParticleSwarm(lower, upper, n_particles),
        options,
    )
    test_summary(res, "Particle Swarm")
    res = Optim.optimize(
        rosenbrock_s,
        initial_x,
        ParticleSwarm(n_particles = n_particles),
        options,
    )
    test_summary(res, "Particle Swarm")

    # c1 and c2 are clamped to [3/2, 5/2] (issue #1203). In states 1 and 4 c2
    # decreases, so starting from 3/2 it must stay at 3/2.
    for state in (1, 4)
        w, c1, c2 = Optim.update_swarm_params!(3 / 2, 3 / 2, 0.9, state, 0.5)
        @test c2 == 3 / 2
        @test 3 / 2 <= c1 <= 5 / 2
    end
end
