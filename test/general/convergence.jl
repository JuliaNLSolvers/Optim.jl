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
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true, false)
    # f_increase
    f0, f1 = 1.0, 1.0 + 1e-7
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true, true)
    # f_increase without convergence
    f_tol = 1e-12
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, false, true, true, true)

    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, false, true, true, true)

    f_tol = 1e-6 # rel tol
    dOpt = DummyOptions(x_tol, f_tol, g_tol, g_abstol)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true, true)

    f0, f1 = 1.0, 1.0 - 1e-7
    dOpt = DummyOptions(x_tol, f_tol, g_tol, g_abstol)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) == (true, true, true, true, false)

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

    @test Optim.initial_convergence(d, ds, dm, ones(2), dOpt) == false
    @test Optim.initial_convergence(d, ds, dm, zeros(2), dOpt) == true

    # Zeroth order methods have no gradient -> returns false by default
    ds = DummyStateZeroth(x1, x0, f1, f0, g)
    dm = DummyMethodZeroth()

    @test Optim.gradient_convergence_assessment(ds,d,dOpt) == false
    @test Optim.initial_convergence(d, ds, dm, ones(2), dOpt) == false

    # should check all other methods as well

end
