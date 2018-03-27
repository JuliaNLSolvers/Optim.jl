@testset "Manifolds" begin
    srand(0)

    # Test case: find eigenbasis for first two eigenvalues of a symmetric matrix by minimizing the Rayleigh quotient under orthogonality constraints
    n = 4
    m = 2
    A = Diagonal(range(1,stop=2,length=n))
    fmanif(x) = real(vecdot(x,A*x)/2)
    gmanif(x) = A*x
    gmanif!(stor,x) = copyto!(stor,gmanif(x))
    # A[2,2] /= 10 #optional: reduce the gap to make the problem artificially harder
    x0 = randn(n,m)+im*randn(n,m)

    manif = Optim.Stiefel()

    # AcceleratedGradientDescent should be compatible also, but I haven't been able to make it converge
    for method in (Optim.GradientDescent, Optim.ConjugateGradient, Optim.LBFGS, Optim.BFGS,
                   Optim.NGMRES, Optim.OACCEL)
        debug_printing && print_with_color(:green, "Solver: $(summary(method()))\n")
        res = Optim.optimize(fmanif, gmanif!, x0, method(manifold=manif))
        debug_printing && print_with_color(:green, "Iter\tf-calls\tg-calls\n")
        debug_printing && print_with_color(:red, "$(Optim.iterations(res))\t$(Optim.f_calls(res))\t$(Optim.g_calls(res))\n")
        @test Optim.converged(res)
    end
    res = Optim.optimize(fmanif, gmanif!, x0, Optim.MomentumGradientDescent(mu=0.0, manifold=manif))
    @test Optim.converged(res)

    # Power
    @views fprod(x) = fmanif(x[:,1]) + fmanif(x[:,2])
    @views gprod!(stor,x) = (gmanif!(stor[:, 1],x[:, 1]);gmanif!(stor[:, 2],x[:, 2]);stor)
    m1 = Optim.PowerManifold(Optim.Sphere(), (n,), (2,))
    srand(0)
    x0 = randn(n,2) + im*randn(n,2)
    res = Optim.optimize(fprod, gprod!, x0, Optim.ConjugateGradient(manifold=m1))
    @test Optim.converged(res)
    minpow = Optim.minimizer(res)

    # Product
    @views fprod(x) = fmanif(x[1:n]) + fmanif(x[n+1:2n])
    @views gprod!(stor,x) = (gmanif!(stor[1:n],x[1:n]);gmanif!(stor[n+1:2n],x[n+1:2n]);stor)
    m2 = Optim.ProductManifold(Optim.Sphere(), Optim.Sphere(), (n,), (n,))
    srand(0)
    x0 = randn(2n) + im*randn(2n)
    res = Optim.optimize(fprod, gprod!, x0, Optim.ConjugateGradient(manifold=m2))
    @test Optim.converged(res)
    minprod = Optim.minimizer(res)

    # results should be exactly equal: same initial guess, same sequence of operations
    @test minpow[:,1] == minprod[1:n]
    @test minpow[:,2] == minprod[n+1:2n]
end
