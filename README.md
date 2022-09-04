Optim.jl
========

Univariate and multivariate optimization in Julia.

Optim.jl is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers) family.

For direct contact to the maintainer, you can reach out directly to pkofod on [slack](https://julialang.org/slack/).

| **Documentation**  | **Build Status** | **Social** | **Reference to cite** |
|:-:|:-:|:-:|:-:|
| [![][docs-stable-img]][docs-stable-url]  | [![Build Status][build-linux-img]][build-linux-url] | [![][gitter-img]][gitter-url]| [![JOSS][joss-img]][joss-url] |
|  |[![Build Status][build-mac-img]][build-mac-url] |  |  |
|  |[![Build Status][build-windows-img]][build-windows-url] |  |  |
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

* Status: success

* Candidate solution
  Minimizer: [1.00e+00, 1.00e+00]
  Minimum:   5.471433e-17

* Found with
  Algorithm:     BFGS
  Initial Point: [0.00e+00, 0.00e+00]

* Convergence measures
  |x - x'|               = 3.47e-07 ≰ 0.0e+00
  |x - x'|/|x'|          = 3.47e-07 ≰ 0.0e+00
  |f(x) - f(x')|         = 6.59e-14 ≰ 0.0e+00
  |f(x) - f(x')|/|f(x')| = 1.20e+03 ≰ 0.0e+00
  |g(x)|                 = 2.33e-09 ≤ 1.0e-08

* Work counters
  Seconds run:   0  (vs limit Inf)
  Iterations:    16
  f(x) calls:    53
  ∇f(x) calls:   53
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

The package is a registered package, and can be installed with `Pkg.add`.

```julia
julia> using Pkg; Pkg.add("Optim")
```
or through the `pkg` REPL mode by typing
```
] add Optim
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

[build-linux-img]: https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/linux.yml/badge.svg
[build-linux-url]: https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/linux.yml

[build-windows-img]: https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/windows.yml/badge.svg
[build-windows-url]: https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/windows.yml

[build-mac-img]: https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/mac.yml/badge.svg
[build-mac-url]: https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/mac.yml

[cov-img]: https://img.shields.io/codecov/c/github/JuliaNLSolvers/Optim.jl/master.svg?maxAge=2592000
[cov-url]: https://codecov.io/gh/JuliaNLSolvers/Optim.jl

[gitter-url]: https://gitter.im/JuliaNLSolvers/Optim.jl
[gitter-img]: https://badges.gitter.im/JuliaNLSolvers/Optim.jl.svg

[zenodo-url]: https://zenodo.org/badge/latestdoi/3933868
[zenodo-img]: https://zenodo.org/badge/3933868.svg

[joss-url]: https://doi.org/10.21105/joss.00615
[joss-img]: http://joss.theoj.org/papers/10.21105/joss.00615/status.svg
