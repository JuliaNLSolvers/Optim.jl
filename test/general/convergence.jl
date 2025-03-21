mutable struct DummyState <: Optim.AbstractOptimizerState
    x
    x_previous
    f_x
    f_x_previous
    g
end

mutable struct DummyStateZeroth <: Optim.ZerothOrderState
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
    g_abstol
end

mutable struct DummyMethod <: Optim.AbstractOptimizer end
mutable struct DummyMethodZeroth <: Optim.ZerothOrderOptimizer end

@testset "Convergence assessment" begin

    ## assess_convergence

    # should converge
    x0, x1 = [1.], [1.0 - 1e-7]
    f0, f1 = 1.0, 1.0 - 1e-7
    g = [1e-7]
    x_tol = 1e-6
    f_tol = 1e-6 # rel tol
    g_tol = 1e-6
    g_abstol = 1e-6
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, false)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, [g_tol]) == (true, true, true, false)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol/1000) == (true, true, false, false)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, [g_tol/1000]) == (true, true, false, false)
    # f_increase
    f0, f1 = 1.0, 1.0 + 1e-7
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true)
    # f_increase without convergence
    f_tol = 1e-12
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, false, true, true)

    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, false, true, true)

    # Implementation with both abstol and reltol
    f_tol = 1e-6
    @test Optim.assess_convergence(x1, x0, f1, f0, g, 0, x_tol, 0, f_tol, g_tol) == (true, true, true, true)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, 0, x_tol, 0, f_tol, [g_tol]) == (true, true, true, true)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, 0, f_tol, 0, g_tol) == (true, true, true, true)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, 0, f_tol, 0, [g_tol]) == (true, true, true, true)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, 0, x_tol/1000, 0, f_tol/1000, g_tol/1000) == (false, false, false, true)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, 0, x_tol/1000, 0, f_tol/1000, [g_tol/1000]) == (false, false, false, true)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol/1000, 0, f_tol/1000, 0, g_tol/1000) == (false, false, false, true)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol/1000, 0, f_tol/1000, 0, [g_tol/1000]) == (false, false, false, true)

    f_tol = 1e-6 # rel tol
    dOpt = DummyOptions(x_tol, f_tol, g_tol, g_abstol)    # FIXME: this isn't used?
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true)

    f0, f1 = 1.0, 1.0 - 1e-7
    dOpt = DummyOptions(x_tol, f_tol, g_tol, g_abstol)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, false)

    ## initial_convergence and gradient_convergence_assessment

    ds = DummyState(x1, x0, f1, f0, g)
    dOpt = DummyOptions(x_tol, f_tol, g_tol, g_abstol)
    dm = DummyMethod()

    # >= First Order
    d = Optim.OnceDifferentiable(x->sum(abs2.(x)),zeros(2))

    Optim.gradient!(d,ones(2))
    @test Optim.gradient_convergence_assessment(ds,d,dOpt) == false
    Optim.gradient!(d,zeros(2))
    @test Optim.gradient_convergence_assessment(ds,d,dOpt) == true

    @test Optim.initial_convergence(d, ds, dm, ones(2), dOpt) == (false, false)
    @test Optim.initial_convergence(d, ds, dm, zeros(2), dOpt) == (true, false)

    # Zeroth order methods have no gradient -> returns false by default
    ds = DummyStateZeroth(x1, x0, f1, f0, g)
    dm = DummyMethodZeroth()

    @test Optim.gradient_convergence_assessment(ds,d,dOpt) == false
    @test Optim.initial_convergence(d, ds, dm, ones(2), dOpt) == (false, false)

    # should check all other methods as well

end
