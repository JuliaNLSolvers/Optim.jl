srand(0)

# Test case: find eigenbasis for first two eigenvalues of a symmetric matrix by minimizing the Rayleigh quotient under orthogonality constraints
n = 4
m = 2
A = Diagonal(linspace(1,2,n))
f(x) = vecdot(x,A*x)/2
g(x) = A*x
g!(stor,x) = copy!(stor,g(x))
# A[2,2] /= 10 #optional: reduce the gap to make the problem artificially harder
x0 = randn(n,m)

manif = Optim.Stiefel()

# AcceleratedGradientDescent should be compatible also, but I haven't been able to make it converge
for method in (Optim.GradientDescent, Optim.ConjugateGradient, Optim.LBFGS, Optim.BFGS)
    res = Optim.optimize(f, g!, x0, method(manifold=manif))
    @test Optim.converged(res)
end
res = Optim.optimize(f, g!, x0, Optim.MomentumGradientDescent(mu=0.0, manifold=manif))
@test Optim.converged(res)

# test product and power manifold
@views fprod(x) = f(x[1:n]) + f(x[n+1:2n])
@views gprod!(stor,x) = (g!(stor[1:n],x[1:n]);g!(stor[n+1:2n],x[n+1:2n]);stor)
m1 = Optim.PowerManifold(Optim.Sphere(), (n,), 2)
m2 = Optim.ProductManifold(Optim.Sphere(), Optim.Sphere(), (n,), (n,))

x0 = randn(2n)
for m in (m1,m2)
    res = Optim.optimize(fprod, gprod!, x0, Optim.ConjugateGradient(manifold=m))
    @test Optim.converged(res)
end
