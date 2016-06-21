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

Instead of using gradient information, we keep track of the function value at a number
of points in the search space. Together, the points form a simplex. Given a simplex,
we can perform one of four actions: reflect, expand, contract, or shrink. Basically,
the goal is to iteratively replace the worst point with a better point. More information
can be found in Nelder and Mead (1965) or Gao and Han (2010).

The stopping rule is the same as in the original paper, and is basically the standard
error of the function values at the vertices. To set the tolerance level for this
convergence criterion, set the `g_tol` level as described in the Configurable Options
section.

When the solver finishes, we return a minimizer which is either the centroid or one of the vertices.
The function value at the centroid adds a function evaluation, as we need to evaluate the objection
at the centroid to choose the smallest function value. Howeever, even if the function value at the centroid can be returned
as the minimum, we do not trace it during the optimization iterations. This is to avoid
too many evaluations of the objective function which can be computationally expensive.
Typically, there should be no more than twice as many `f_calls` than `iterations`,
and adding an evaluation at the centroid when `tracing` could considerably increase the total
run-time of the algorithm.

## Example

## References
Nelder, John A. and R. Mead (1965). "A simplex method for function minimization". Computer Journal 7: 308–313. doi:10.1093/comjnl/7.4.308.
Gao, Fuchang and Lixing Han (2010). "Implementing the Nelder-Mead simplex algorithm with adaptive parameters". Computational Optimization and Applications [DOI 10.1007/s10589-010-9329-3]
