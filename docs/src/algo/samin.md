# SAMIN
## Constructor
```julia
SAMIN(; nt::Int = 5     # reduce temperature every nt*ns*dim(x_init) evaluations
        ns::Int = 5     # adjust bounds every ns*dim(x_init) evaluations
        rt::T = 0.9     # geometric temperature reduction factor: when temp changes, new temp is t=rt*t
        neps::Int = 5   # number of previous best values the final result is compared to
        f_tol::T = 1e-12 # the required tolerance level for function value comparisons
        x_tol::T = 1e-6 # the required tolerance level for x
        coverage_ok::Bool = false, # if false, increase temperature until initial parameter space is covered
        verbosity::Int = 0) # scalar: 0, 1, 2 or 3 (default = 0).
```
## Description
The `SAMIN` method implements the Simulated Annealing algorithm for problems with
bounds constraints as described in Goffe et. al. (1994) and Goffe (1996). A key control
parameter is rt, the geometric temperature reduction rate, which should be between zero
and one. Setting rt lower will cause the algorithm to contract the search space more quickly,
reducing the run time. Setting rt too low will cause the algorithm to narrow the search
too quickly, and the true minimizer may be skipped over. If possible, run the algorithm
multiple times to verify that the same solution is found each time. If this is not the case,
increase rt. When in doubt, start with a conservative rt, for example, rt=0.95, and allow for
a generous iteration limit. The algorithm requires lower and upper bounds on the parameters,
although these bounds are often set rather wide, and are not necessarily meant to reflect
constraints in the model, but rather bounds that enclose the parameter space. If the final
`x`s are very close to the boundary (which can be checked by setting verbosity=1), it is a
good idea to restart the optimizer with wider bounds, unless the bounds actually reflect
hard constraints on `x`.

## Example
This example shows a successful minimization:
```julia
julia> using Optim, OptimTestProblems

julia> prob = OptimTestProblems.UnconstrainedProblems.examples["Rosenbrock"];

julia> res = Optim.optimize(prob.f, fill(-100.0, 2), fill(100.0, 2), prob.initial_x, SAMIN(), Optim.Options(iterations=10^6))
================================================================================
SAMIN results
==> Normal convergence <==
total number of objective function evaluations: 23701

     Obj. value:      0.0000000000

       parameter      search width
         1.00000           0.00000
         1.00000           0.00000
================================================================================

Results of Optimization Algorithm
 * Algorithm: SAMIN
 * Starting Point: [-1.2,1.0]
 * Minimizer: [0.9999999893140956,0.9999999765350857]
 * Minimum: 5.522977e-16
 * Iterations: 23701
 * Convergence: false
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = NaN
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = NaN |f(x)|
   * |g(x)| ≤ 0.0e+00: false
     |g(x)| = NaN
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 23701
 * Gradient Calls: 0
```
## Example
This example shows an unsuccessful minimization, because the cooling rate,
rt=0.5, is too rapid:
```julia
julia> using Optim, OptimTestProblems

julia> prob = OptimTestProblems.UnconstrainedProblems.examples["Rosenbrock"];
julia> res = Optim.optimize(prob.f, fill(-100.0, 2), fill(100.0, 2), prob.initial_x, SAMIN(rt=0.5), Optim.Options(iterations=10^6))
================================================================================
SAMIN results
==> Normal convergence <==
total number of objective function evaluations: 12051

     Obj. value:      0.0011613045

       parameter      search width
         0.96592           0.00000
         0.93301           0.00000
================================================================================

Results of Optimization Algorithm
 * Algorithm: SAMIN
 * Starting Point: [-1.2,1.0]
 * Minimizer: [0.9659220825756248,0.9330054696322896]
 * Minimum: 1.161304e-03
 * Iterations: 12051
 * Convergence: false
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = NaN
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = NaN |f(x)|
   * |g(x)| ≤ 0.0e+00: false
     |g(x)| = NaN
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 12051
 * Gradient Calls: 0

```

## References
 - Goffe, et. al. (1994) "Global Optimization of Statistical Functions with Simulated Annealing", Journal of Econometrics, V. 60, N. 1/2.
 - Goffe, William L. (1996) "SIMANN: A Global Optimization Algorithm using Simulated Annealing " Studies in Nonlinear Dynamics & Econometrics, Oct96, Vol. 1 Issue 3.
