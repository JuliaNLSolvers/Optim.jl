Optim.jl
========

Univariate and multivariate optimization in Julia.

| **Documentation**                                         | **PackageEvaluator**                      |**Build Status** |**Social**                                 |
|:---------------------------------------------------------:|:-----------------------------------------:|:---:|:------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url]  | [![][pkg-0.4-img]][pkg-0.4-url]| [![Build Status][build-img]][build-url] | [![][gitter-img]][gitter-url]|
| [![][docs-latest-img]][docs-latest-url]  | [![][pkg-0.5-img]][pkg-0.5-url]| [![Codecov branch][cov-img]][cov-url] |  |

# Optimization

Optim.jl is a package for univariate and multivariate optimization of functions.
A typical example of the usage of Optim.jl is
```julia
using Optim
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
result = optimize(rosenbrock, zeros(2), BFGS())
```
Which gives the output
```jlcon
Results of Optimization Algorithm
 * Algorithm: BFGS
 * Starting Point: [0.0,0.0]
 * Minimizer: [0.9999999929485311,0.9999999859278653]
 * Minimum: 4.981810e-17
 * Iterations: 21
 * Convergence: true
   * |x - x'| < 1.0e-32: false
   * |f(x) - f(x')| / |f(x)| < 1.0e-32: true
   * |g(x)| < 1.0e-08: false
   * Reached Maximum Number of Iterations: false
 * Objective Function Calls: 157
 * Gradient Calls: 157
```
For more details and options, see the documentation ([stable](https://juliaopt.github.io/Optim.jl/stable) | [latest](https://juliaopt.github.io/Optim.jl/latest)).

# Installation

The package is registered in `METADATA.jl` and can be installed with `Pkg.add`.

```julia
julia> Pkg.add("Optim")
```

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://juliaopt.github.io/Optim.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliaopt.github.io/Optim.jl/stable

[build-img]: https://travis-ci.org/JuliaOpt/Optim.jl.svg?branch=master
[build-url]: https://travis-ci.org/JuliaOpt/Optim.jl

[pkg-0.4-img]: http://pkg.julialang.org/badges/Optim_0.4.svg
[pkg-0.4-url]: http://pkg.julialang.org/?pkg=Optim&ver=0.4
[pkg-0.5-img]: http://pkg.julialang.org/badges/Optim_0.5.svg
[pkg-0.5-url]: http://pkg.julialang.org/?pkg=Optim&ver=0.5

[cov-img]: https://img.shields.io/codecov/c/github/JuliaOpt/Optim.jl/master.svg?maxAge=2592000
[cov-url]: https://codecov.io/gh/JuliaOpt/Optim.jl

[gitter-url]: https://gitter.im/JuliaOpt/Optim.jl
[gitter-img]: https://badges.gitter.im/JuliaOpt/Optim.jl.svg
