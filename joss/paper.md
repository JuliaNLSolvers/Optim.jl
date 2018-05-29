---
title: 'Optim: A mathematical optimization package for Julia'
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
[Optim](https://github.com/JuliaNLSolvers/Optim.jl/) provides a range
of optimization capabilities written in the Julia programming language
[@bezanson2017julia]. Our aim is to enable researchers, users, and
other Julia packages to solve optimization problems without writing
such algorithms themselves.
The package supports optimization on manifolds,
functions of complex numbers, and input types such as arbitrary
precision vectors and matrices.  We have implemented routines for
derivative free, first-order, and second-order optimization methods.
The user can provide derivatives themselves, or request that they are
calculated using automatic differentiation or finite difference
methods.  The main focus of the package has currently been on
unconstrained optimization, however, box-constrained optimization is
supported, and a more comprehensive support for constraints is
underway.

Similar to Optim, the C library
[NLopt](http://ab-initio.mit.edu/nlopt) [@johnson2018nlopt] contains a
collection of nonlinear optimization routines. In Python,
[scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)
supports many of the same algorithms as Optim does, and
[Pymanopt](https://pymanopt.github.io/) [@townsend2016pymanopt] is a
toolbox for manifold optimization.
Within the Julia community, the packages
[BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl)
and
[Optimize.jl](https://github.com/JuliaSmoothOptimizers/Optimize.jl)
provide optimization capabilities focusing on derivative-free and
large-scale smooth problems respectively.
The packages [Convex.jl](https://github.com/JuliaOpt/Convex.jl) and
[JuMP.jl](https://github.com/JuliaOpt/JuMP.jl) [@dunning2017jump] define
modelling languages for which users can formulate optimization problems.
In contrast to the previously mentioned optimization codes, Convex and JuMP
work as abstraction layers between the user and solvers from a other packages.

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
further improve upon.  We would also like to thank everyone who has
contributed with code and discussions to help improve the package.  In
particular, Antoine Levitt, Christoph Ortner, and Chris Rackauckas
have been helpful in providing suggestions and code contributions
towards more modularity and greater support for non-trivial inputs and
decision spaces.

# Funding
Asbjørn Riseth is partially supported by the EPSRC research grant EP/L015803/1.

# References
