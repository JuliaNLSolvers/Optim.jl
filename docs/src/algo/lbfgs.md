# (L-)BFGS

This page contains information about
Broyden–Fletcher–Goldfarb–Shanno ([BFGS](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm)) algorithm and its limited memory version [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS).

## Constructors

```julia
BFGS(; alphaguess = LineSearches.InitialStatic(),
       linesearch = LineSearches.HagerZhang(),
       initial_invH = nothing,
       initial_stepnorm = nothing,
       manifold = Flat())
```

`initial_invH` has a default value of `nothing`. If the user has a specific initial
matrix they want to supply, it should be supplied as a function of an array similar
to the initial point `x0`.

If `initial_stepnorm` is set to a number `z`, the initial matrix will be the
identity matrix scaled by `z` times the sup-norm of the gradient at the initial
point `x0`.

```julia
LBFGS(; m = 10,
        alphaguess = LineSearches.InitialStatic(),
        linesearch = LineSearches.HagerZhang(),
        P = nothing,
        precondprep = (P, x) -> nothing,
        manifold = Flat(),
        scaleinvH0::Bool = P === nothing)
```

## Description

In both algorithms the aim is do compute a descent direction ``d_ n``
by approximately solving the newton equation

```math
H_n d_n = - ∇f(x_n),
```

where ``H_n`` is an approximation to the Hessian of ``f``. Instead of approximating
the Hessian, both BFGS as well as L-BFGS approximate the inverse ``B_n = H_n^{-1}`` of the Hessian,
since that yields a matrix multiplication instead of solving a the linear system of equations above.

Then

```math
x_{n+1} = x_n - \alpha_n d_n,
```

where ``α_n`` is the step size resulting from the specified `linesearch`.

In (L-)BFGS, the matrix is an approximation to the inverse of the Hessian built using differences of the gradients and iterates during the iterations.
As long as the initial matrix is positive definite it is possible to show that all the follow matrices will be as well.

For BFGS, the starting matrix could simply be the identity matrix, such that the first step is identical
to the Gradient Descent algorithm, or even the actual inverse of the initial Hessian.
While BFGS stores the full matrix ``B_n`` and performs an update of that approximate Hessian in every step.

L-BFGS on the other hand only stores ``m`` differences of gradients and iterates
instead of a full matrix. This is more memory-efficient especially for large-scale problems.

For L-BFGS, the inverse of the Hessian can be preconditioned in two ways.

You can either set `scaleinvH0` to true, then the `m` steps of approximating
the inverse of the Hessian start from a scaled version of the identity.
It if is set to false, the approximation starts from the identity matrix.

On the other hand you can provide a preconditioning matrix `P` that should be positive definite the approximation then starts from ``P^{-1}``.
The preconditioner can be changed during the iterations by providing the `precondprep` keyword which based on `P` and the current iterate `x` updates
the preconditioner matrix accordingly.

## References

```@bibliography
Pages = []
Canonical = false

nocedal2006
```
