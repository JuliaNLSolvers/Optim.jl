mutable struct DummyState <: Optim.AbstractOptimizerState
    x::Any
    x_previous::Any
    f_x::Any
    f_x_previous::Any
    g_x::Any
end

mutable struct DummyStateZeroth <: Optim.ZerothOrderState
    x::Any
    x_previous::Any
    f_x::Any
    f_x_previous::Any
    g_x::Any
end

mutable struct DummyOptions
    x_tol::Any
    f_tol::Any
    g_tol::Any
    g_abstol::Any
end

mutable struct DummyMethod <: Optim.AbstractOptimizer end
mutable struct DummyMethodZeroth <: Optim.ZerothOrderOptimizer end

@testset "Convergence assessment" begin

    ## assess_convergence

    # should converge
    x0, x1 = [1.0], [1.0 - 1e-7]
    f0, f1 = 1.0, 1.0 - 1e-7
    g = [1e-7]
    x_tol = 1e-6
    f_tol = 1e-6 # rel tol
    g_tol = 1e-6
    g_abstol = 1e-6
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) ==
          (true, true, true, false)
    # f_increase
    f0, f1 = 1.0, 1.0 + 1e-7
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) ==
          (true, true, true, true)
    # f_increase without convergence
    f_tol = 1e-12
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) ==
          (true, false, true, true)

    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) ==
          (true, false, true, true)

    f_tol = 1e-6 # rel tol
    dOpt = DummyOptions(x_tol, f_tol, g_tol, g_abstol)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) ==
          (true, true, true, true)

    f0, f1 = 1.0, 1.0 - 1e-7
    dOpt = DummyOptions(x_tol, f_tol, g_tol, g_abstol)
    @test Optim.assess_convergence(x1, x0, f1, f0, g, x_tol, f_tol, g_tol) ==
          (true, true, true, false)

    ## initial_convergence and gradient_convergence_assessment

    ds = DummyState(x1, x0, f1, f0, g)
    opt = Optim.Options(; x_abstol = x_tol, f_abstol = f_tol, g_tol, g_abstol)
    dm = DummyMethod()

    # >= First Order
    d = Optim.OnceDifferentiable(x -> sum(abs2.(x)), zeros(2))

    x = ones(2)
    f_x, g_x = Optim.value_gradient!(d, x)
    ds = DummyState(x, x0, f_x, f0, g_x)
    @test !Optim.gradient_convergence_assessment(ds, opt)
    @test Optim.initial_convergence(ds, opt) == (false, false)
    x = zeros(2)
    f_x, g_x = Optim.value_gradient!(d, x)
    ds = DummyState(x, x0, f_x, f0, g_x)
    @test Optim.gradient_convergence_assessment(ds, opt)
    @test Optim.initial_convergence(ds, opt) == (true, false)

    # Zeroth order methods have no gradient -> returns false by default
    ds = DummyStateZeroth(x1, x0, f1, f0, g)
    dm = DummyMethodZeroth()

    x = ones(2)
    f_x, g_x = Optim.value_gradient!(d, x)
    ds = DummyState(x, x0, f_x, f0, g_x)
    @test !Optim.gradient_convergence_assessment(ds, opt)
    @test Optim.initial_convergence(ds, opt) == (false, false)

    # should check all other methods as well

end
