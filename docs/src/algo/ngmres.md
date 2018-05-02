# Acceleration methods: N-GMRES and O-ACCEL
## Constructors
```julia
NGMRES(;
        alphaguess = LineSearches.InitialStatic(),
        linesearch = LineSearches.HagerZhang(),
        manifold = Flat(),
        wmax::Int = 10,
        ϵ0 = 1e-12,
        nlprecon = GradientDescent(
            alphaguess = LineSearches.InitialStatic(alpha=1e-4,scaled=true),
            linesearch = LineSearches.Static(),
            manifold = manifold),
        nlpreconopts = Options(iterations = 1, allow_f_increases = true),
      )
```

```julia
OACCEL(;manifold::Manifold = Flat(),
       alphaguess = LineSearches.InitialStatic(),
       linesearch = LineSearches.HagerZhang(),
       nlprecon = GradientDescent(
           alphaguess = LineSearches.InitialStatic(alpha=1e-4,scaled=true),
           linesearch = LineSearches.Static(),
           manifold = manifold),
       nlpreconopts = Options(iterations = 1, allow_f_increases = true),
       ϵ0 = 1e-12,
       wmax::Int = 10)
```

## Description
These algorithms take a step given by the nonlinear preconditioner `nlprecon`
and proposes an accelerated step on a subspace spanned by the previous
`wmax` iterates.

- N-GMRES accelerates based on a minimization of an approximation to the $\ell_2$ norm of the
gradient.
- O-ACCEL accelerates based on a minimization of a n approximation to the objective.

N-GMRES was originally developed for solving nonlinear systems [1], and reduces to
GMRES for linear problems.
Application of the algorithm to optimization is covered, for example, in [2].
A description of O-ACCEL and its connection to N-GMRES can be found in [3].

*We recommend trying [LBFGS](lbfgs.md) on your problem before N-GMRES or O-ACCEL. All three algorithms have similar computational cost and memory requirements, however, L-BFGS is more efficient for many problems.*

## Example

This example shows how to accelerate `GradientDescent` on the Extended Rosenbrock problem.
First, we try to optimize using `GradientDescent`.

```julia
using Optim, OptimTestProblems
UP = OptimTestProblems.UnconstrainedProblems
prob = UP.examples["Extended Rosenbrock"]
optimize(UP.objective(prob), UP.gradient(prob), prob.initial_x, GradientDescent())
```
The algorithm does not converge within 1000 iterations.

```
Results of Optimization Algorithm
 * Algorithm: Gradient Descent
 * Starting Point: [-1.2,1.0, ...]
 * Minimizer: [0.8923389282461412,0.7961268644300445, ...]
 * Minimum: 2.898230e-01
 * Iterations: 1000
 * Convergence: false
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 4.02e-04
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 2.38e-03 |f(x)|
   * |g(x)| ≤ 1.0e-08: false
     |g(x)| = 8.23e-02
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: true
 * Objective Calls: 2525
 * Gradient Calls: 2525
```

Now, we use `OACCEL` to accelerate `GradientDescent`.
```julia
# Default nonlinear procenditioner for `OACCEL`
nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
                           linesearch=LineSearches.Static())
# Default size of subspace that OACCEL accelerates over is `wmax = 10`
oacc10 = OACCEL(nlprecon=nlprecon, wmax=10)
optimize(UP.objective(prob), UP.gradient(prob), prob.initial_x, oacc10)
```
This drastically improves the `GradientDescent` algorithm, converging in 87 iterations.
```
Results of Optimization Algorithm
 * Algorithm: O-ACCEL preconditioned with Gradient Descent
 * Starting Point: [-1.2,1.0, ...]
 * Minimizer: [1.0000000011361219,1.0000000022828495, ...]
 * Minimum: 3.255053e-17
 * Iterations: 87
 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 6.51e-08
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 7.56e+02 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 1.06e-09
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 285
 * Gradient Calls: 285
```

We can improve the acceleration further by changing the acceleration subspace size `wmax`.
```julia
oacc5 = OACCEL(nlprecon=nlprecon, wmax=5)
optimize(UP.objective(prob), UP.gradient(prob), prob.initial_x, oacc5)
```
Now, the O-ACCEL algorithm has accelerated `GradientDescent` to converge in 50 iterations.
```
Results of Optimization Algorithm
 * Algorithm: O-ACCEL preconditioned with Gradient Descent
 * Starting Point: [-1.2,1.0, ...]
 * Minimizer: [0.9999999999392858,0.9999999998784691, ...]
 * Minimum: 9.218164e-20
 * Iterations: 50
 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 2.76e-07
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 5.18e+06 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 4.02e-11
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 181
 * Gradient Calls: 181
```

As a final comparison, we can do the same with N-GMRES.
```julia
ngmres5 = NGMRES(nlprecon=nlprecon, wmax=5)
optimize(UP.objective(prob), UP.gradient(prob), prob.initial_x, ngmres5)
```
Again, this significantly improves the `GradientDescent` algorithm, and converges in 63 iterations.
```
Results of Optimization Algorithm
 * Algorithm: Nonlinear GMRES preconditioned with Gradient Descent
 * Starting Point: [-1.2,1.0, ...]
 * Minimizer: [0.9999999998534468,0.9999999997063993, ...]
 * Minimum: 5.375569e-19
 * Iterations: 63
 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 9.94e-09
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 1.29e+03 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 4.94e-11
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 222
 * Gradient Calls: 222
```

## References
[1] De Sterck. Steepest descent preconditioning for nonlinear GMRES optimization. NLAA, 2013.
[2] Washio and Oosterlee. Krylov subspace acceleration for nonlinear multigrid schemes. ETNA, 1997.
[3] Riseth. Objective acceleration for unconstrained optimization. 2018.
