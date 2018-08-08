# Manifold optimization
Optim.jl supports the minimization of functions defined on Riemannian manifolds, i.e. with simple constraints such as normalization and orthogonality. The basic idea of such algorithms is to project back ("retract") each iterate of an unconstrained minimization method onto the manifold. This is used by passing a `manifold` keyword argument to the optimizer.

## Howto
Here is a simple test case where we minimize the Rayleigh quotient `<x, A x>` of a symmetric matrix `A` under the constraint `||x|| = 1`, finding an eigenvector associated with the lowest eigenvalue of `A`.
```julia
n = 10
A = Diagonal(range(1, stop=2, length=n))
f(x) = dot(x,A*x)/2
g(x) = A*x
g!(stor,x) = copyto!(stor,g(x))
x0 = randn(n)

manif = Optim.Sphere()
Optim.optimize(f, g!, x0, Optim.ConjugateGradient(manifold=manif))
```

## Supported solvers and manifolds
All first-order optimization methods are supported.

The following manifolds are currently supported:
* Flat: Euclidean space, default. Standard unconstrained optimization.
* Sphere: spherical constraint `||x|| = 1`
* Stiefel: Stiefel manifold of N by n matrices with orthogonal columns, i.e. `X'*X = I`

The following meta-manifolds construct manifolds out of pre-existing ones:
* PowerManifold: identical copies of a specified manifold
* ProductManifold: product of two (potentially different) manifolds

See `test/multivariate/manifolds.jl` for usage examples.

Implementing new manifolds is as simple as adding methods `project_tangent!(M::YourManifold,x)` and `retract!(M::YourManifold,g,x)`. If you implement another manifold or optimization method, please contribute a PR!

## References
The Geometry of Algorithms with Orthogonality Constraints, Alan Edelman, Tomás A. Arias, Steven T. Smith, SIAM. J. Matrix Anal. & Appl., 20(2), 303–353

Optimization Algorithms on Matrix Manifolds, P.-A. Absil, R. Mahony, R. Sepulchre, Princeton University Press, 2008
