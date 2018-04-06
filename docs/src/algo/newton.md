# Newton's Method
## Constructor
```julia
Newton(; alphaguess = LineSearches.InitialStatic(),
         linesearch = LineSearches.HagerZhang())
```

The constructor takes two keywords:

* `linesearch = a(d, x, p, x_new, g_new, phi0, dphi0, c)`, a function performing line search, see the line search section.
* `alphaguess = a(state, dphi0, d)`, a function for setting the initial guess for the line search algorithm, see the line search section.

## Description
Newton's method for optimization has a long history, and is in some sense the
gold standard in unconstrained optimization of smooth functions, at least from a theoretical viewpoint.
The main benefit is that it has a quadratic rate of convergence near a local optimum. The main
disadvantage is that the user has to provide a Hessian. This can be difficult, complicated, or simply annoying.
It can also be computationally expensive to calculate it.

Newton's method for optimization consists of applying Newton's method for solving
systems of equations, where the equations are the first order conditions, saying
that the gradient should equal the zero vector.

```math
\nabla f(x) = 0
```

A second order Taylor expansion of the left-hand side leads to the iterative scheme

```math
x_{n+1} = x_n - H(x_n)^{-1}\nabla f(x_n)
```

where the inverse is not calculated directly, but the step size is instead calculated by solving

```math
H(x) \textbf{s} = \nabla f(x_n).
```

This is equivalent to minimizing a quadratic model, ``m_k`` around the current ``x_n``

```math
m_k(s) = f(x_n) + \nabla f(x_n)^\top \textbf{s} + \frac{1}{2} \textbf{s}^\top H(x_n) \textbf{s}
```

For functions where ``H(x_n)`` is difficult, or computationally expensive to obtain, we might
replace the Hessian with another positive definite matrix that approximates it.
Such methods are called Quasi-Newton methods; see (L-)BFGS and Gradient Descent.

In a sufficiently small neighborhood around the minimizer, Newton's method has
quadratic convergence, but globally it might have slower convergence, or it might
even diverge. To ensure convergence, a line search is performed for each ``\textbf{s}``.
This amounts to replacing the step formula above with

```math
x_{n+1} = x_n - \alpha \textbf{s}
```

and finding a scalar ``\alpha`` such that we get sufficient descent; see the line search section for more information.

Additionally, if the function is locally
concave, the step taken in the formulas above will go in a direction of ascent,
 as the Hessian will not be positive (semi)definite.
To avoid this, we use a specialized method to calculate the step direction. If
the Hessian is positive semidefinite then the method used is standard, but if
it is not, a correction is made using the functionality in [PositiveFactorizations.jl](https://github.com/timholy/PositiveFactorizations.jl).

## Example
show the example from the issue

## References
