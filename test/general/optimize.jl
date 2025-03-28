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
    initial_invH = zeros(2, 2)
    h1(initial_invH, [127.0, 921.0])
    initial_invH = Matrix(Diagonal(diag(initial_invH)))
    results = optimize(
        f1,
        g1,
        [127.0, 921.0],
        BFGS(initial_invH = x -> initial_invH),
        Optim.Options(),
    )
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

    # Tests for PR #302
    results = optimize(cos, 0, 2pi)
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0.0, 2pi)
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0, 2pi, Brent())
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0.0, 2pi, Brent())
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
    opt = Optim.Options(store_trace = true, trace_simplex = true, extended_trace = true)
    res = optimize(f, x0, NelderMead(), opt)
    tr = Optim.simplex_trace(res)
    trval = Optim.simplex_value_trace(res)
    trcent = Optim.centroid_trace(res)
    @test tr[end] != tr[end-1]
    @test trval[end] != trval[end-1]
    @test trcent[end] != trcent[end-1]
end

@testset "Test NaN Termination Tolerance" begin
    rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

    # Need high-precision gradient to impose tight tolerances below
    function g_rosenbrock!(g, x)
        g[1] = 2 * (1.0 - x[1]) * (-1) + 2 * 100 * (x[2] - x[1]^2) * (-2 * x[1])
        g[2] = 2 * 100 * (x[2] - x[1]^2)
    end

    # To set tight tolerance on gradient g, need to disable any check on f
    options = Optim.Options(g_tol = 1e-10, f_reltol = NaN, f_abstol = NaN)
    result = Optim.optimize(
        rosenbrock,
        g_rosenbrock!,
        zeros(2),
        Optim.ConjugateGradient(),
        options,
    )
    @test Optim.g_residual(result) < 1e-10

    # To set tight tolerance on x, need to also disable default gradient tolerance, g_tol=1e-8
    options = Optim.Options(x_tol = 1e-10, g_tol = NaN, f_reltol = NaN, f_abstol = NaN)
    result = Optim.optimize(
        rosenbrock,
        g_rosenbrock!,
        zeros(2),
        Optim.ConjugateGradient(),
        options,
    )
    @test Optim.x_abschange(result) < 1e-10
end
