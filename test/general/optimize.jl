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

    # Tests for PR #302
    results = optimize(cos, 0, 2pi);
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0.0, 2pi);
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0, 2pi, Brent());
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0.0, 2pi, Brent());
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0, 2pi, method = Brent())
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0.0, 2pi, method = Brent())
    @test norm(Optim.minimizer(results) - pi) < 0.01
end


@testset "nm trace" begin
    # https://github.com/JuliaNLSolvers/Optim.jl/issues/1112
    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

    x0 = [0.0, 0.0]
    opt = Optim.Options(store_trace = true,
                        trace_simplex = true,
                        extended_trace = true)
    res = optimize(f, x0, NelderMead(), opt)
    tr = Optim.simplex_trace(res)
    trval = Optim.simplex_value_trace(res)
    trcent = Optim.centroid_trace(res)
    @test tr[end] != tr[end-1]
    @test trval[end] != trval[end-1]
    @test trcent[end] != trcent[end-1]
end