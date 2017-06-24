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
res = Optim.optimize(fcomplex, gcomplex!, x0, Optim.ConjugateGradient())
@test Optim.converged(res)
@test Optim.minimizer(res) ≈ A\b rtol=1e-2
