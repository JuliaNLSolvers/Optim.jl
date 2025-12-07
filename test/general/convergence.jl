mutable struct DummyState{TX,TF,TG} <: Optim.AbstractOptimizerState
    x::TX
    x_previous::TX
    f_x::TF
    f_x_previous::TF
    g_x::TG
end

mutable struct DummyStateZeroth{TX,TF} <: Optim.ZerothOrderState
    x::TX
    x_previous::TX
    f_x::TF
    f_x_previous::TF
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
    ds = DummyStateZeroth(x1, x0, f1, f0)
    dm = DummyMethodZeroth()

    x = ones(2)
    f_x, g_x = Optim.value_gradient!(d, x)
    ds = DummyState(x, x0, f_x, f0, g_x)
    @test !Optim.gradient_convergence_assessment(ds, opt)
    @test Optim.initial_convergence(ds, opt) == (false, false)

    # should check all other methods as well

    for T in (Float32, Float64)
        ds = DummyState(T[-1.3, 2.5, -4.1], T[-1.1, 2.8, -4.0], f_x, f0, zeros(3))
        @test @inferred(Optim.x_abschange(ds))::T ≈ 0.3
        @test iszero((s -> @allocated(Optim.x_abschange(s)))(ds))
        @test @inferred(Optim.x_relchange(ds))::T ≈ 0.3 / 4.1
        @test iszero((s -> @allocated(Optim.x_relchange(s)))(ds))

        # Special case: Empty state
        ds = DummyState(T[], T[], f_x, f0, empty(g_x))
        @test iszero(@inferred(Optim.x_abschange(ds))::T)
        @test iszero((s -> @allocated(Optim.x_abschange(s)))(ds))
        @test isnan(@inferred(Optim.x_relchange(ds))::T)
        @test iszero((s -> @allocated(Optim.x_relchange(s)))(ds))
    end
end
