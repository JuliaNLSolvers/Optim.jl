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
end
