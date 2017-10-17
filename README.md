Optim.jl
========

Univariate and multivariate optimization in Julia.

| **Documentation**                                         | **PackageEvaluator**                      |**Build Status** |**Social**                                 |
|:---------------------------------------------------------:|:-----------------------------------------:|:---:|:------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url]  | [![][pkg-0.4-img]][pkg-0.4-url]| [![Build Status][build-img]][build-url] | [![][gitter-img]][gitter-url]|
| [![][docs-latest-img]][docs-latest-url]  | [![][pkg-0.5-img]][pkg-0.5-url]|[![Build Status][winbuild-img]][winbuild-url] |  |
| |[![][pkg-0.6-img]][pkg-0.6-url] | [![Codecov branch][cov-img]][cov-url]  ||

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

Which gives the output
```jlcon
Results of Optimization Algorithm
 * Algorithm: BFGS
 * Starting Point: [0.0,0.0]
 * Minimizer: [0.9999999926033423,0.9999999852005353]
 * Minimum: 5.471433e-17
 * Iterations: 16
 * Convergence: true
   * |x - x'| < 1.0e-32: false 
     |x - x'| = 3.47e-07 
   * |f(x) - f(x')| / |f(x)| < 1.0e-32: false
     |f(x) - f(x')| / |f(x)| = NaN 
   * |g(x)| < 1.0e-08: true 
     |g(x)| = 2.33e-09 
   * stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 53
 * Gradient Calls: 53
```
For more details and options, see the documentation ([stable](https://julianlsolvers.github.io/Optim.jl/stable) | [latest](https://julianlsolvers.github.io/Optim.jl/latest)).

# Installation

The package is registered in `METADATA.jl` and can be installed with `Pkg.add`.

```julia
julia> Pkg.add("Optim")
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
