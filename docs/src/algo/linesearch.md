# Line search
## Description

The line search functionality has been moved to
[LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl).

Line search is used to decide the step length along the direction computed by
an optimization algorithm.

The following `Optim` algorithms use line search:
* Accelerated Gradient Descent
* (L-)BFGS
* Conjugate Gradient
* Gradient Descent
* Momentum Gradient Descent
* Newton

By default `Optim` calls the line search algorithm `HagerZhang()` provided by `LineSearches`.
Different line search algorithms can be assigned with
the `linesearch` keyword argument to the given algorithm.

`LineSearches` also allows the user to decide how the
initial step length for the line search algorithm is chosen.
This is set with the `alphaguess` keyword argument for the `Optim` algorithm.
The default procedure varies.


## Example
This example compares two different line search algorithms on the Rosenbrock problem.

First, run `Newton` with the default line search algorithm:
```julia
using Optim, LineSearches
prob = Optim.UnconstrainedProblems.examples["Rosenbrock"]

algo_hz = Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.HagerZhang())
res_hz = Optim.optimize(prob.f, prob.g!, prob.h!, prob.initial_x, method=algo_hz)
```

This gives the result
``` julia
 * Algorithm: Newton's Method
 * Starting Point: [0.0,0.0]
 * Minimizer: [0.9999999999999994,0.9999999999999989]
 * Minimum: 3.081488e-31
 * Iterations: 14
 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 3.06e-09
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 2.94e+13 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 1.11e-15
   * stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 44
 * Gradient Calls: 44
 * Hessian Calls: 14
```

Now we can try `Newton` with the More-Thuente line search:
``` julia
algo_mt = Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.MoreThuente())
res_mt = Optim.optimize(prob.f, prob.g!, prob.h!, prob.initial_x, method=algo_mt)
```

This gives the following result, reducing the number of function and gradient calls:
``` julia
Results of Optimization Algorithm
 * Algorithm: Newton's Method
 * Starting Point: [0.0,0.0]
 * Minimizer: [0.9999999999999992,0.999999999999998]
 * Minimum: 2.032549e-29
 * Iterations: 14
 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 3.67e-08
   * |f(x) - f(x')| ≤ 0.0e00 |f(x)|: false
     |f(x) - f(x')| = 1.66e+13 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 1.76e-13
   * stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 17
 * Gradient Calls: 17
 * Hessian Calls: 14
```

## References
