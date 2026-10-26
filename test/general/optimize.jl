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
    options = Optim.Options(g_abstol = 1e-10, f_reltol = NaN, f_abstol = NaN)
    result = Optim.optimize(
        rosenbrock,
        g_rosenbrock!,
        zeros(2),
        Optim.ConjugateGradient(),
        options,
    )
    @test Optim.g_residual(result) < 1e-10

    # To set tight tolerance on x, need to also disable default gradient tolerance, g_tol=1e-8
    options = Optim.Options(x_abstol = 1e-10, g_abstol = NaN, f_reltol = NaN, f_abstol = NaN)
    result = Optim.optimize(
        rosenbrock,
        g_rosenbrock!,
        zeros(2),
        Optim.ConjugateGradient(),
        options,
    )
    @test Optim.x_abschange(result) < 1e-10
end
