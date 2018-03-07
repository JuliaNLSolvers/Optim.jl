---
title: 'Optim: An optimization package for Julia'
tags:
  - Optimization
  - Julia
authors:
  - name: Patrick K Mogensen
    orcid: 0000-0002-4910-1932
    affiliation: 1
  - name: Asbjørn N Riseth
    orcid: 0000-0002-5861-7885
    affiliation: 2
affiliations:
  - name: University of Copenhagen
    index: 1
  - name: University of Oxford
    index: 2
date: 7 March 2018
bibliography: paper.bib
---

# Summary
`Optim.jl` provides a range of optimization capabilities written in
the Julia programming language [@bezanson2017julia].  The package
supports optimization on manifolds, functions of complex numbers, and
input types such as arbitrary precision vectors and matrices.  We have
implemented routines for derivative free, first-order, and
second-order optimization methods.  The user can provide derivatives
themselves, or request that they are calculated using automatic
differentiation or finite difference methods.  The main focus of the
package has currently been on unconstrained optimization, however,
box-constrained optimization is supported, and a more comprehensive
support for constraints is underway.

## Optimization routines
As of version 0.14, the following optimization routines are available.

- Second-order methods
    * Newton
    * Newton with trust region
    * Hessian-vector with trust region
- First-order methods
    * BFGS
    * L-BFGS (with linear preconditioning)
    * Conjugate gradient (with linear preconditioning)
    * Gradient descent (with linear preconditioning)
- Acceleration methods
    * Nonlinear GMRES
    * Objective acceleration
- Derivative-free methods
    * Nelder–Mead
    * Simulated annealing
    * Particle swarm
- Interval bound univariate methods
    * Brent's method
    * Golden-section search

The derivative based methods use line searches to assist
convergence. Multiple line search algorithms are available, including
interpolating backtracking and methods that aim to satisfy the Wolfe
conditions.

## Example usage

We provide short example of how one can use Optim to minimize the
Rosenbrock function, defined by

$$
f(x,y) = {(1-x)}^2 + 100\times {(y-x^2)}^2.
$$

In the following code, we define the objective and tell Optim to
minimize the objective using BFGS with a second-order backtracking
line search.

``` julia
using Optim, LineSearches
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x0 = zeros(2)
result = optimize(rosenbrock, x0, BFGS(linesearch = BackTracking(order=2))
```

When the gradient is not provided, the default behaviour is to
approximate derivatives using finite differences. The summary output
from the optimization is stored in `result`, and prints

``` julia
Results of Optimization Algorithm
 * Algorithm: BFGS
 * Starting Point: [0.0,0.0]
 * Minimizer: [0.9999999926655744,0.9999999853309254]
 * Minimum: 5.379380e-17
 * Iterations: 23
 * Convergence: true
   * |x - x'| ≤ 1.0e-32: false
     |x - x'| = 1.13e-09
   * |f(x) - f(x')| ≤ 1.0e-32 |f(x)|: false
     |f(x) - f(x')| = 1.57e-01 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 8.79e-11
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 31
 * Gradient Calls: 24
```

# Usage in research and industry
The optimization routines in this package have been used in both
industrial and academic contexts.  For example, parts of the internal
work in the company Ternary Intelligence Inc. [@ternary2017] rely on
the package.  Notably, an upcoming book on optimization
[@mykel2018optimization] uses Optim for its examples.  Optim has been
used for a wide range of applications in academic research, including
optimal control [@riseth2017comparison; @riseth2017dynamic], parameter
estimation [@riseth2017operator; @rackauckas2017differentialequations;
and @dony2018parametric], quantum physics [@damle2018variational],
crystalline modelling [@chen2017qm; @braun2017effect], and
the large-scale astronomical cataloguing project Celeste
[@regier2015celeste; @regier2016celeste].  A new acceleration scheme
for optimization [@riseth2017objective], and a preconditioning scheme
for geometry optimisation [@packwood2016universal]
have also been tested within the Optim framework.



# Acknowledgements
John Myles White initiated the development of the Optim code base
in 2012.  We owe much to him and Timothy Holy for creating a solid
package for optimization that the rest of the Julia community could
further improve upon.  We would also like to thank everyone that have
contributed with code and discussions to help improve the package.  In
particular, Antoine Levitt, Christoph Ortner, and Chris Rackauckas
have been helpful in providing suggestions and code contributions
towards more modularity and greater support for non-trivial inputs and
decision spaces.

# Funding
Asbjørn Riseth is partially supported by the EPSRC research grant EP/L015803/1.

# References
