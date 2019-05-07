# (L-)BFGS
This page contains information about BFGS and its limited memory version L-BFGS.
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
        scaleinvH0::Bool = true && (typeof(P) <: Nothing))
```
## Description
This means that it takes steps according to

```math
x_{n+1} = x_n - P^{-1}\nabla f(x_n)
```

where ``P`` is a positive definite matrix. If ``P`` is the Hessian, we get Newton's method.
In (L-)BFGS, the matrix is an approximation to the Hessian built using differences
in the gradient across iterations. As long as the initial matrix is positive definite
 it is possible to show that all the follow matrices will be as well. The starting
matrix could simply be the identity matrix, such that the first step is identical
to the Gradient Descent algorithm, or even the actual Hessian.

There are two versions of BFGS in the package: BFGS, and L-BFGS. The latter is different
from the former because it doesn't use a complete history of the iterative procedure to
construct ``P``, but rather only the latest ``m`` steps. It doesn't actually build the Hessian
approximation matrix either, but computes the direction directly. This makes more suitable for
large scale problems, as the memory requirement to store the relevant vectors will
grow quickly in large problems.

As with the other quasi-Newton solvers in this package, a scalar ``\alpha`` is introduced
as follows

```math
x_{n+1} = x_n - \alpha P^{-1}\nabla f(x_n)
```

and is chosen by a linesearch algorithm such that each step gives sufficient descent.

## Example
## References
Wright, Stephen, and Jorge Nocedal (2006) "Numerical optimization." Springer
