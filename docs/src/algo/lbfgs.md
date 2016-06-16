# (L-)BFGS
## Constructor
```julia
LBFGS(; m = 10,
        linesearch! = hz_linesearch!,
        P = nothing,
        precondprep! = (P, x) -> nothing)
```
## Description
This means that it takes steps according to

$ x_{n+1} = x_n - P^{-1}\nabla f(x_n)$

where $P$ is a positive definite matrix. If $P$ is the Hessian, we get Newton's method.
In (L-)BFGS, the matrix is an approximation to the Hessian built using differences
in the gradient across iterations. As long as the initial matrix is positive definite,
then it is possible to show that all consecutive matrices will be as well. The starting
matrix could simply be the identity matrix, such that the first step is identical
to the Gradient Descent algorithm, or even the actual Hessian.

There are two versions of BFGS in the package: BFGS, and L-BFGS. The latter is different
from the former because it doesn't store the approximated Hessian directly in dense form.
This makes it suitable for large scale problems.
## Example
## References
