# L-BFGS-B
## Constructor
```julia
LBFGSB(; m::Integer = 10,
         linesearch = HZAW(),
         stepsize = 1.0,
         clip_subspace::Bool = true)
```
The keyword `m` sets how many `(s, y)` correction pairs are stored in the
limited-memory Hessian approximation. `linesearch` selects one of the line
searches bundled with the solver (`HZAW()`, the Hager-Zhang approximate-Wolfe
search, or `MTLS()`, the Moré-Thuente search); these are self-contained and do
not depend on LineSearches.jl. `stepsize` is the default trial step handed to
the line search, and `clip_subspace` switches between Fortran-style
component-wise clamping of the subspace step (`true`) and the paper's
proportional backtracking (`false`).

## Description
`LBFGSB` solves problems of the form

```math
\min_{x} f(x) \quad \text{subject to} \quad l \le x \le u
```

where the bounds `l` and `u` may be finite, infinite, or equal (a fixed
variable). It is the limited-memory BFGS algorithm with box constraints of
Byrd, Lu, Nocedal and Zhu (1995). Unlike `Fminbox`, which wraps an
unconstrained solver inside a log-barrier loop, `LBFGSB` handles the bounds
natively and is a bound-constrained optimizer in its own right (a sibling of
`Fminbox` and `SAMIN`).

Each iteration

1. computes the *generalized Cauchy point* by following the projected gradient
   and stopping at the first set of bounds that minimizes the limited-memory
   quadratic model,
2. minimizes that quadratic model over the variables left free at the Cauchy
   point, and
3. performs a line search along the resulting search direction, capped at the
   distance to the nearest bound so that every iterate stays feasible.

The Hessian approximation is stored in the compact representation
``B = \theta I - W M W^\top``, which keeps the per-iteration cost linear in the
number of variables for a fixed memory length `m`.

Convergence is assessed on the infinity norm of the projected gradient
``\|x - P_{[l,u]}(x - g)\|_\infty`` against `Options.g_abstol`, in addition to
the usual change-in-`x` and change-in-`f` criteria. This projected-gradient
norm, not the plain ``\|g\|_\infty``, is what the results summary reports on the
`|g(x)|` line; the two coincide only when no bound is active. Tolerances, the
iteration cap, the time limit, and callbacks are all taken from `Optim.Options`.

## Example
`LBFGSB` is called with lower and upper bounds, which may be given as scalars or
as arrays. The starting point must be feasible (`l .<= x0 .<= u`).

```julia-repl
julia> using Optim, OptimTestProblems

julia> prob = UnconstrainedProblems.examples["Rosenbrock"];

julia> optimize(prob.f, fill(-2.0, 2), fill(2.0, 2), prob.initial_x, LBFGSB())
 * Status: success

 * Candidate solution
    Final objective value:     5.021329e-17

 * Found with
    Algorithm:     L-BFGS-B

 * Convergence measures
    |x - x'|               = 2.67e-08 ≰ 0.0e+00
    |x - x'|/|x'|          = 2.67e-08 ≰ 0.0e+00
    |f(x) - f(x')|         = 4.94e-23 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 9.84e-07 ≰ 0.0e+00
    |g(x)|                 = 2.34e-09 ≤ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    40
    f(x) calls:    99
    ∇f(x) calls:   99
    ∇f(x)ᵀv calls: 0
```

Scalar bounds are broadcast to every coordinate, so the same problem on the box
``[-2, 2]^2`` can also be written `optimize(prob.f, -2.0, 2.0, prob.initial_x, LBFGSB())`.

## References

```@bibliography
byrd1995
```
