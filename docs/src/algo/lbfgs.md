# (L-)BFGS

This page contains information about
Broyden–Fletcher–Goldfarb–Shanno ([BFGS](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm)) algorithm and its limited memory version [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS).

## Constructors

```julia
BFGS(; alphaguess = LineSearches.InitialStatic(),
       linesearch = LineSearches.HagerZhang(),
       initial_invH = nothing,
       initial_stepnorm = 1.0,
       manifold = Flat(),
       initial_scaling = Optim.ShannoPhua(),
       self_scaling = Optim.AlBaali())
```

`initial_invH` has a default value of `nothing`. If the user has a specific initial
matrix they want to supply, it should be supplied as a function of an array similar
to the initial point `x0`.

If `initial_stepnorm` is set to a number `z`, the initial matrix will be the
identity matrix divided by the sup-norm of the gradient at the initial point `x0`
and multiplied by `z`, so that the first trial step has sup-norm `z`. The default is
`z = 1`; `initial_stepnorm = nothing` keeps the identity.

The three scaling options are on by default. `BFGS(initial_stepnorm = nothing,
initial_scaling = nothing, self_scaling = nothing)` is the unscaled method that was the
default up to Optim 2.

With `initial_scaling = Optim.ShannoPhua()` and no `initial_invH`, the initial matrix is
replaced by ``(s^T y / y^T y) I`` just before the first update and after any reset of
the search direction (Shanno and Phua, 1978). `self_scaling` scales the Hessian
approximation ``B`` by ``\tau`` before every other update, with ``\rho = y^T s / s^T B s``:
`Optim.OrenLuenberger()` uses ``\tau = \rho`` (Oren and Luenberger, 1974) and
`Optim.AlBaali(; cutoff = 0.5)` uses ``\tau = \min(1, \rho)``, unscaled when
``\rho`` is below the cutoff (Al-Baali, 1998).

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

In both algorithms the aim is to compute a descent direction ``d_ n``
by approximately solving the Newton equation

```math
H_n d_n = - ∇f(x_n),
```

where ``H_n`` is an approximation to the Hessian of ``f``. Instead of approximating
the Hessian, both BFGS as well as L-BFGS approximate the inverse ``B_n = H_n^{-1}`` of the Hessian,
since that yields a matrix multiplication instead of requiring that we solve the linear system of
equations above.

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

You can either set `scaleinvH0` to `true`, then the `m` steps of approximating
the inverse of the Hessian start from a scaled version of the identity.
If it is set to `false`, the approximation starts from the identity matrix.

Alternatively, you can provide a positive definite preconditioning matrix `P`; the approximation then starts from ``P^{-1}``.
The preconditioner can be changed during the iterations by providing the `precondprep` keyword which, based on `P` and the current iterate `x`, updates
the preconditioner matrix accordingly.

## References

```@bibliography
Pages = []
Canonical = false

nocedal2006
shanno1978
oren1974
albaali1998
```
