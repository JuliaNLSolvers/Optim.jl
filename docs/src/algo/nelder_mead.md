# Nelder-Mead
Nelder-Mead is currently the standard algorithm when no derivatives are provided.
## Constructor
```julia
NelderMead(; a = 1.0,
             g = 2.0,
             b = 0.5)
```
## Description
Our current implementation of the Nelder-Mead algorithm follows the original implementation
very closely, see Nelder and Mead (1965). This means that there is scope for improvement, but
also that it should be quite clear what is going on in the code relative to the original paper.

The stopping rule is the same as in the original paper, and is basically the standard
error of the function values at the vertices. This means that setting the `f_tol` keyword
does not put a restriction on `f()` exactly, but on this standard error measure instead.

Upon (non-)convergence, we return a minimizer which is either the centroid or one of the vertices.
This adds a function evaluation, as we choose the minimizer according to point with
the smallest function value. Even if the function value at the centroid can be returned
as the minimum, we do not trace it during the optimization iterations. This is to avoid
too many evaluations of `f()` which can be expensive for many applications of the Nelder-Mead
algorithm. Typically, there should be no more than twice as many `f_calls` than `iterations`,
and adding an evaluation at the centroid when `tracing` could considerably increase the total
run-time of the algorithm.

## Example

## References
Nelder, John A.; R. Mead (1965). "A simplex method for function minimization". Computer Journal 7: 308–313. doi:10.1093/comjnl/7.4.308.
