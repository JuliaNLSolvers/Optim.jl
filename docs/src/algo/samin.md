# SAMIN
## Constructor
```julia
SAMIN(; nt::Int = 5     # reduce temperature every nt*ns*dim(x_init) evaluations
        ns::Int = 5     # adjust bounds every ns*dim(x_init) evaluations
        rt::T = 0.5     # geometric temperature reduction factor: when temp changes, new temp is t=rt*t
        neps::Int = 5   # number of previous best values the final result is compared to
        f_tol::T = 1e-8 # the required tolerance level for function value comparisons
        x_tol::T = 1e-5 # the required tolerance level for x
        coverage_ok::Bool = false, # increase temperature until parameter space is covered
        verbosity::Int = 0) # scalar: 0, 1, 2 or 3 (default = 0).
```
## Description
The `SAMIN` method implements the Simulated Annealing algorithm for problems with
bounds constrains a described in Goffe et. al. (1994) and Goffe (1996). The
algorithm requires bounds, although these bounds are often set rather wide, and
are not necessarily meant to reflect constraints in the model, but rather bounds
that are there to improve the search. If the final `x`s are very close to the boundary,
it is a good idea to restart the optimizer with wider bounds, unless the bounds
actually reflect hard constraints on `x`.

## Example
```julia
julia> using Optim, OptimTestProblems

julia> prob = OptimTestProblems.UnconstrainedProblems.examples["Rosenbrock"];

julia> res = Optim.optimize(prob.f, prob.initial_x, fill(-100.0, 2), fill(100.0, 2), SAMIN(), Optim.Options(iterations=20000))
Results of Optimization Algorithm
 * Algorithm: SAMIN
 * Starting Point: [-1.2,1.0]
 * Minimizer: [1.0170012560033472,1.0343750729202243]
 * Minimum: 2.897402e-04
 * Iterations: 5901
 * Convergence: false
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = NaN
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = NaN |f(x)|
   * |g(x)| ≤ 0.0e+00: false
     |g(x)| = NaN
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 5901
 * Gradient Calls: 0
```

## References
 - Goffe, et. al. (1994) "Global Optimization of Statistical Functions with Simulated Annealing", Journal of Econometrics, V. 60, N. 1/2.
 - Goffe, William L. (1996) "SIMANN: A Global Optimization Algorithm using Simulated Annealing " Studies in Nonlinear Dynamics & Econometrics, Oct96, Vol. 1 Issue 3.
