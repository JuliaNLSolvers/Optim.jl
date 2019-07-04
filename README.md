Optim.jl
========

Univariate and multivariate optimization in Julia.

Optim.jl is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers) family.

| **Documentation**  | **Build Status** | **Social** | **Reference to cite** |
|:-:|:-:|:-:|:-:|
| [![][docs-stable-img]][docs-stable-url]  | [![Build Status][build-img]][build-url] | [![][gitter-img]][gitter-url]| [![JOSS][joss-img]][joss-url] |
|  |[![Build Status][winbuild-img]][winbuild-url] |  |  |
| |[![Codecov branch][cov-img]][cov-url]  || |

# Optimization

Optim.jl is a package for univariate and multivariate optimization of functions.
A typical example of the usage of Optim.jl is
```julia
using Optim
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
result = optimize(rosenbrock, zeros(2), BFGS())
```
This minimizes the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)

<img src="https://user-images.githubusercontent.com/8431156/31627324-2bbc9ebc-b2ad-11e7-916f-857ad8dcb714.gif" title="f(x,y) = (a-x)^2+b(y-x^2)^2" />

with a = 1, b = 100 and the initial values x=0, y=0.
The minimum is at (a,a^2).

The above code gives the output
```jlcon
Results of Optimization Algorithm
 * Algorithm: BFGS
 * Starting Point: [0.0,0.0]
 * Minimizer: [0.9999999926033423,0.9999999852005353]
 * Minimum: 5.471433e-17
 * Iterations: 16

 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 3.47e-07
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 1.20e+03 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 2.33e-09
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 53
 * Gradient Calls: 53
```
To get information on the keywords used to construct method instances, use the Julia REPL help prompt (`?`)
```
help?> LBFGS
search: LBFGS

     LBFGS
    ≡≡≡≡≡≡≡

     Constructor
    =============

  LBFGS(; m::Integer = 10,
  alphaguess = LineSearches.InitialStatic(),
  linesearch = LineSearches.HagerZhang(),
  P=nothing,
  precondprep = (P, x) -> nothing,
  manifold = Flat(),
  scaleinvH0::Bool = true && (typeof(P) <: Nothing))

  LBFGS has two special keywords; the memory length m, and
  the scaleinvH0 flag. The memory length determines how many
  previous Hessian approximations to store. When scaleinvH0
  == true, then the initial guess in the two-loop recursion
  to approximate the inverse Hessian is the scaled identity,
  as can be found in Nocedal and Wright (2nd edition) (sec.
  7.2).

  In addition, LBFGS supports preconditioning via the P and
  precondprep keywords.

     Description
    =============

  The LBFGS method implements the limited-memory BFGS
  algorithm as described in Nocedal and Wright (sec. 7.2,
  2006) and original paper by Liu & Nocedal (1989). It is a
  quasi-Newton method that updates an approximation to the
  Hessian using past approximations as well as the gradient.

     References
    ============

    •    Wright, S. J. and J. Nocedal (2006), Numerical
        optimization, 2nd edition. Springer

    •    Liu, D. C. and Nocedal, J. (1989). "On the
        Limited Memory Method for Large Scale
        Optimization". Mathematical Programming B. 45
        (3): 503–528
```

# Documentation
For more details and options, see the documentation
- [STABLE][docs-stable-url] — most recently tagged version of the documentation.
- [LATEST][docs-latest-url] — in-development version of the documentation.

# Installation

The package is registered in `METADATA.jl` and can be installed with `Pkg.add`.

```julia
julia> Pkg.add("Optim")
```

# Citation

If you use `Optim.jl` in your work, please cite the following.

```tex
@article{mogensen2018optim,
  author  = {Mogensen, Patrick Kofod and Riseth, Asbj{\o}rn Nilsen},
  title   = {Optim: A mathematical optimization package for {Julia}},
  journal = {Journal of Open Source Software},
  year    = {2018},
  volume  = {3},
  number  = {24},
  pages   = {615},
  doi     = {10.21105/joss.00615}
}
```

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://julianlsolvers.github.io/Optim.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://julianlsolvers.github.io/Optim.jl/stable

[build-img]: https://travis-ci.org/JuliaNLSolvers/Optim.jl.svg?branch=master
[build-url]: https://travis-ci.org/JuliaNLSolvers/Optim.jl

[winbuild-img]: https://ci.appveyor.com/api/projects/status/prp8ygfp4rr9tafe?svg=true
[winbuild-url]: https://ci.appveyor.com/project/blegat/optim-jl

[pkg-0.4-img]: http://pkg.julialang.org/badges/Optim_0.4.svg
[pkg-0.4-url]: http://pkg.julialang.org/?pkg=Optim&ver=0.4
[pkg-0.5-img]: http://pkg.julialang.org/badges/Optim_0.5.svg
[pkg-0.5-url]: http://pkg.julialang.org/?pkg=Optim&ver=0.5
[pkg-0.6-img]: http://pkg.julialang.org/badges/Optim_0.6.svg
[pkg-0.6-url]: http://pkg.julialang.org/?pkg=Optim&ver=0.6

[cov-img]: https://img.shields.io/codecov/c/github/JuliaNLSolvers/Optim.jl/master.svg?maxAge=2592000
[cov-url]: https://codecov.io/gh/JuliaNLSolvers/Optim.jl

[gitter-url]: https://gitter.im/JuliaNLSolvers/Optim.jl
[gitter-img]: https://badges.gitter.im/JuliaNLSolvers/Optim.jl.svg

[zenodo-url]: https://zenodo.org/badge/latestdoi/3933868
[zenodo-img]: https://zenodo.org/badge/3933868.svg

[joss-url]: https://doi.org/10.21105/joss.00615
[joss-img]: http://joss.theoj.org/papers/10.21105/joss.00615/status.svg
