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
        debug_printing && print_with_color(:green, "Solver: $(summary(method()))\n")
        res = Optim.optimize(fcomplex, gcomplex!, x0, method())
        debug_printing && print_with_color(:green, "Iter\tf-calls\tg-calls\n")
        debug_printing && print_with_color(:red, "$(Optim.iterations(res))\t$(Optim.f_calls(res))\t$(Optim.g_calls(res))\n")
        @test Optim.converged(res)
        @test Optim.minimizer(res) ≈ A\b rtol=1e-2
    end

    debug_printing && print_with_color(:green, "Solver: $(summary(MomentumGradientDescent(mu=0.0)))\n")
    res = Optim.optimize(fcomplex, gcomplex!, x0, Optim.MomentumGradientDescent(mu=0.0))
    debug_printing && print_with_color(:green, "Iter\tf-calls\tg-calls\n")
    debug_printing && print_with_color(:red, "$(Optim.iterations(res))\t$(Optim.f_calls(res))\t$(Optim.g_calls(res))\n")
    @test Optim.converged(res)
    @test Optim.minimizer(res) ≈ A\b rtol=1e-2
end
