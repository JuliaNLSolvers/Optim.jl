# Gradient Descent
## Constructor
```julia
GradientDescent(; linesearch!::Function = hz_linesearch!,
                  P = nothing,
                  precondprep! = (P, x) -> nothing)
```
## Description
Gradient Descent a common name for a quasi-Newton solver. This means that it takes
steps according to

$ x_{n+1} = x_n - P^{-1}\nabla f(x_n)$

where $P$ is the Hessian in Newton's method, but is generally just a positive definite
matrix. In Gradient Descent, $P$ is simply an appropriately dimensioned identity matrix.
This means that we go in the exact opposite direction of the gradient. This means
that we do not use the curvature information from the Hessian, or an approximation
of it. While it does seem quite logical to go in the opposite direction of the fastest
increase in objective value, the procedure can be very slow if the problem is ill-conditioned.
See the section on preconditioners for ways to remedy this when using Gradient Descent.
## Example
## References
