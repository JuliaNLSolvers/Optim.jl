@testset "Complex numbers" begin
    srand(0)

    # Test case: solve Ax=b with A and b complex
    n = 4
    A = randn(n,n) + im*randn(n,n)
    A = A'A + I
    b = randn(n) + im*randn(n)

    fcomplex(x) = real(vecdot(x,A*x)/2 - vecdot(b,x))
    gcomplex(x) = A*x-b
    gcomplex!(stor,x) = copy!(stor,gcomplex(x))
    x0 = randn(n)+im*randn(n)

    @testset "Finite difference" begin
        oda1 = OnceDifferentiable(fcomplex, x0)
        fx = NLSolversBase.value_gradient!(oda1, x0)
        @test fx == fcomplex(x0)
        @test gcomplex(x0) ≈ NLSolversBase.gradient(oda1)
    end

    for method in (Optim.NelderMead,Optim.ParticleSwarm,Optim.Newton)
        @test_throws Any Optim.optimize(fcomplex, gcomplex!, x0, method())
    end
    #not supposed to converge, but it should at least go through without errors
    res = Optim.optimize(fcomplex, gcomplex!, x0, Optim.SimulatedAnnealing())

    # TODO: AcceleratedGradientDescent fail to converge?
    for method in (Optim.GradientDescent, Optim.ConjugateGradient, Optim.LBFGS, Optim.BFGS,
                   Optim.NGMRES, Optim.OACCEL)
        solver = method()
        allow_f_increases = (method in [Optim.GradientDescent,]) # Fix Win32 failure
        dopts = Optim.default_options(solver)
        if haskey(dopts, :allow_f_increases)
            allow_f_increases = allow_f_increases || dopts[:allow_f_increases]
            delete!(dopts, :allow_f_increases)
        end
        options = Optim.Options(allow_f_increases = allow_f_increases; dopts...)

        debug_printing && print_with_color(:green, "Solver: $(summary(solver))\n")
        res = Optim.optimize(fcomplex, gcomplex!, x0, solver,
                             Optim.Options(allow_f_increases=allow_f_increases))
        debug_printing && print_with_color(:green, "Iter\tf-calls\tg-calls\n")
        debug_printing && print_with_color(:red, "$(Optim.iterations(res))\t$(Optim.f_calls(res))\t$(Optim.g_calls(res))\n")
        if !Optim.converged(res)
            warn("$(summary(solver)) failed.")
            display(res)
            println("########################")
        end
        ressum = summary(res) # Just check that no errors arise when doing display(res)
        @test typeof(fcomplex(x0)) == typeof(Optim.minimum(res))
        @test eltype(x0) == eltype(Optim.minimizer(res))
        @test Optim.converged(res)
        @test Optim.minimizer(res) ≈ A\b rtol=1e-2

        @testset "Finite difference" begin
            res = Optim.optimize(fcomplex, x0, solver, options)
            @test Optim.converged(res)
            @test Optim.minimizer(res) ≈ A\b rtol=1e-2
        end
    end

    solver = Optim.MomentumGradientDescent(mu=0.0) # mu = 0 is basically GradienDescent?
    allow_f_increases = true # Fix Win32 failure
    dopts = Optim.default_options(solver)
    if haskey(dopts, :allow_f_increases)
        allow_f_increases = allow_f_increases || dopts[:allow_f_increases]
        delete!(dopts, :allow_f_increases)
    end
    options = Optim.Options(allow_f_increases = allow_f_increases; dopts...)
    debug_printing && print_with_color(:green, "Solver: $(summary(solver))\n")
    res = Optim.optimize(fcomplex, gcomplex!, x0, solver, options)
    debug_printing && print_with_color(:green, "Iter\tf-calls\tg-calls\n")
    debug_printing && print_with_color(:red, "$(Optim.iterations(res))\t$(Optim.f_calls(res))\t$(Optim.g_calls(res))\n")
    if !Optim.converged(res)
        warn("$(summary(solver)) failed.")
        display(res)
        println("########################")
    end
    ressum = summary(res) # Just check that no errors arise when doing display(res)
    @test typeof(fcomplex(x0)) == typeof(Optim.minimum(res))
    @test eltype(x0) == eltype(Optim.minimizer(res))
    @test Optim.converged(res)
    @test Optim.minimizer(res) ≈ A\b rtol=1e-2
    @testset "Finite difference" begin
        res = Optim.optimize(fcomplex, x0, solver, options)
        @test Optim.converged(res)
        @test Optim.minimizer(res) ≈ A\b rtol=1e-2
    end
end
