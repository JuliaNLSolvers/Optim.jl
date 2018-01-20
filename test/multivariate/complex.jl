@testset "Complex numbers" begin
    srand(0)

    # Test case: solve Ax=b with A and b complex
    n = 4
    A = randn(n,n) + im*randn(n,n)
    A = A'A + I
    b = randn(4) + im*randn(4)

    fcomplex(x) = real(vecdot(x,A*x)/2 - vecdot(b,x))
    gcomplex(x) = A*x-b
    gcomplex!(stor,x) = copy!(stor,gcomplex(x))
    x0 = randn(n)+im*randn(n)

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
            display(res)
        end
        ressum = summary(res) # Just check that no errors arise when doing display(res)
        @test typeof(fcomplex(x0)) == typeof(Optim.minimum(res))
        @test eltype(x0) == eltype(Optim.minimizer(res))
        @test Optim.converged(res)
        @test Optim.minimizer(res) ≈ A\b rtol=1e-2
    end

    solver = Optim.MomentumGradientDescent(mu=0.0)
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
    display(res)
    ressum = summary(res) # Just check that no errors arise when doing display(res)
    @test typeof(fcomplex(x0)) == typeof(Optim.minimum(res))
    @test eltype(x0) == eltype(Optim.minimizer(res))
    @test Optim.converged(res)
    @test Optim.minimizer(res) ≈ A\b rtol=1e-2
end
