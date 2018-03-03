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
algorithm

## References
 - Goffe, et. al. (1994) "Global Optimization of Statistical Functions with Simulated Annealing", Journal of Econometrics, V. 60, N. 1/2.
 - Goffe, William L. (1996) "SIMANN: A Global Optimization Algorithm using Simulated Annealing " Studies in Nonlinear Dynamics & Econometrics, Oct96, Vol. 1 Issue 3.
