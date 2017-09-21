mutable struct DummyState
    x
    x_previous
    f_x
    f_x_previous
    g
end

mutable struct DummyOptions
    x_tol
    f_tol
    g_tol
end

@testset "assess_convergence" begin
    # should converge
    x0, x1 = [1.], [1.0 - 1e-7]
    f0, f1 = 1.0, 1.0 - 1e-7
    g = [1e-7]
    x_tol = 1e-6
    f_tol = 1e-6 # rel tol
    g_tol = 1e-6
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true, false)
    # f_increase
    f0, f1 = 1.0, 1.0 + 1e-7
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true, true)
    # f_increase without convergence
    f_tol = 1e-12
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, false, true, true, true)

    ds = DummyState(x1, x0, f1, f0, g)
    dOpt = DummyOptions(x_tol, f_tol, g_tol)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, false, true, true, true)

    f_tol = 1e-6 # rel tol
    dOpt = DummyOptions(x_tol, f_tol, g_tol)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true, true)

    f0, f1 = 1.0, 1.0 - 1e-7
    dOpt = DummyOptions(x_tol, f_tol, g_tol)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true, false)

    # should check all other methods as well
end
