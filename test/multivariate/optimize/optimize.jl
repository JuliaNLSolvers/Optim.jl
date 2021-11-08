@testset "optimize" begin
    eta = 0.9

    function f1(x)
        (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    function g1(storage, x)
        storage[1] = x[1]
        storage[2] = eta * x[2]
    end

    function h1(storage, x)
        storage[1, 1] = 1.0
        storage[1, 2] = 0.0
        storage[2, 1] = 0.0
        storage[2, 2] = eta
    end

    results = optimize(f1, g1, h1, [127.0, 921.0])
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

    results = optimize(f1, g1, [127.0, 921.0])
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

    results = optimize(f1, [127.0, 921.0])
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

    results = optimize(f1, [127.0, 921.0])
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

    # tests for bfgs_initial_invH
    initial_invH = zeros(2,2)
    h1(initial_invH, [127.0, 921.0])
    initial_invH = Matrix(Diagonal(diag(initial_invH)))
    results = optimize(f1, g1, [127.0, 921.0], BFGS(initial_invH = x -> initial_invH), Optim.Options())
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

    # test timeout
    function f2(x)
        sleep(0.1)
        (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    results = optimize(f2, g1, [127.0, 921.0], BFGS(), Optim.Options(; time_limit=0.0))
    @test !Optim.g_converged(results)
    @test Optim.time_limit(results) < Optim.time_run(results)
end

@testset "#718" begin
    
    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    function g!(G, x)
      G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
      G[2] = 200.0 * (x[2] - x[1]^2)
    end
    function h!(H, x)
      H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
      H[1, 2] = -400.0 * x[1]
      H[2, 1] = -400.0 * x[1]
      H[2, 2] = 200.0
    end
    function fg!(F,G,x)
      G === nothing || g!(G,x)
      F === nothing || return f(x)
      nothing
    end
    function fgh!(F,G,H,x)
      G === nothing || g!(G,x)
      H === nothing || h!(H,x)
      F === nothing || return f(x)
      nothing
    end

    optimize(Optim.only_fg!(fg!), [0., 0.], NelderMead())
    optimize(Optim.only_fgh!(fgh!), [0., 0.], NelderMead())
    optimize(Optim.only_fgh!(fgh!), [0., 0.], ParticleSwarm())
    optimize(Optim.only_fgh!(fgh!), [0., 0.], SimulatedAnnealing())
    optimize(Optim.only_fgh!(fgh!), [0., 0.], GradientDescent())
    optimize(Optim.only_fgh!(fgh!), [0., 0.], LBFGS())
    optimize(Optim.only_fgh!(fgh!), [0., 0.], BFGS())
    optimize(Optim.only_fgh!(fgh!), [0., 0.], NewtonTrustRegion())
    optimize(Optim.only_fgh!(fgh!), [0., 0.], Newton())

end
