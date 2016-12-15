# Line search
## Description

The line search functionality has been moved to
[LineSearches.jl](https://github.com/anriseth/LineSearches.jl).

Line search is used to decide the step length along the direction computed by
an optimization algorithm.

The following `Optim` algorithms use line search:
* Accelerated Gradient Descent
* (L-)BFGS
* Conjugate Gradient
* Gradient Descent
* Momentum Gradient Descent
* Newton

By default `Optim` calls the line search algorithm `hagerzhang!` provided by `LineSearches`.
Different line search algorithms can be assigned with
the `linesearch` keyword argument to the given algorithm.

## Example
This example compares two different line search algorithms on the Rosenbrock problem.

First, run `Newton` with the default line search algorithm:
```julia
using Optim, LineSearches
prob = Optim.UnconstrainedProblems.examples["Rosenbrock"]

algo_hz = Newton(;linesearch = LineSearches.hagerzhang!)
res_hz = Optim.optimize(prob.f, prob.g!, prob.h!, prob.initial_x, method=algo_hz)
```

This gives the result
``` julia
Results of Optimization Algorithm
 * Algorithm: Newton's Method
 * Starting Point: [0.0,0.0]
 * Minimizer: [0.9999999999979515,0.9999999999960232]
 * Minimum: 5.639268e-24
 * Iterations: 13
 * Convergence: true
   * |x - x'| < 1.0e-32: false
   * |f(x) - f(x')| / |f(x)| < 1.0e-32: false
   * |g(x)| < 1.0e-08: true
   * Reached Maximum Number of Iterations: false
 * Objective Function Calls: 54
 * Gradient Calls: 54
```

Now we can try `Newton` with the More-Thuente line search:
``` julia
algo_mt = Newton(;linesearch = LineSearches.morethuente!)
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
   * |x - x'| < 1.0e-32: false
   * |f(x) - f(x')| / |f(x)| < 1.0e-32: false
   * |g(x)| < 1.0e-08: true
   * Reached Maximum Number of Iterations: false
 * Objective Function Calls: 31
 * Gradient Calls: 31
```

## References
