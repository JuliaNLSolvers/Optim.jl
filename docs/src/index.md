# Optim.jl

Univariate and multivariate optimization in Julia.

Optim.jl is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers) family.

| **Source**  | **PackageEvaluator** | **Build Status** | **Social** | **References to cite** |
|:-:|:-:|:-:|:-:|:-:|
| [![Source][github-img-url]][github-repo-url] | [![][pkg-0.6-img]][pkg-0.6-url]| [![Build Status][build-img]][build-url] | [![][gitter-img]][gitter-url]| [![JOSS][joss-img]][joss-url] |
| [![Codecov branch][cov-img]][cov-url]  | [![][pkg-0.5-img]][pkg-0.5-url]|[![Build Status][winbuild-img]][winbuild-url] |  | [![DOI][zenodo-img]][zenodo-url] |


## What
Optim is a Julia package for optimizing functions of
various kinds. While there is some support for box constrained and Riemannian optimization, most
of the solvers try to find an ``x`` that minimizes a function ``f(x)`` without any constraints.
Thus, the main focus is on unconstrained optimization.
The provided solvers, under certain conditions, will converge to a local minimum.
In the case where a global minimum is desired, global optimization techniques should be employed instead (see e.g. [BlackBoxOptim](https://github.com/robertfeldt/BlackBoxOptim.jl)).

## Why
There are many solvers available from both free and commercial sources, and many
of them are accessible from Julia. Few of them are written in Julia.
Performance-wise this is rarely a problem, as they are often written in either
Fortran or C. However, solvers written directly in Julia
does come with some advantages.

When writing Julia software (packages) that require something to be optimized, the programmer
can either choose to write their own optimization routine, or use one of the many
available solvers. For example, this could be something from the [NLOpt](https://github.com/JuliaOpt/NLopt.jl) suite.
This means adding a dependency which is not written in Julia, and more assumptions
have to be made as to the environment the user is in. Does the user have the proper
compilers? Is it possible to use GPL'ed code in the project? Optim is released
under the MIT license, and installation is a simple `Pkg.add`, so it really doesn't
get much freer, easier, and lightweight than that.

It is also true, that using a solver written in C or Fortran makes it impossible to leverage one
of the main benefits of Julia: multiple dispatch. Since Optim is entirely written
in Julia, we can currently use the dispatch system to ease the use of custom preconditioners.
A planned feature along these lines is to allow for user controlled choice of solvers
for various steps in the algorithm, entirely based on dispatch, and not predefined
possibilities chosen by the developers of Optim.

Being a Julia package also means that Optim has access to the automatic differentiation
features through the packages in [JuliaDiff](http://www.juliadiff.org/).

## How
Optim is registered in [METADATA.jl](https://github.com/JuliaLang/METADATA.jl).
This means that all you need to do to install Optim, is to run
```julia
Pkg.add("Optim")
```

## But...
Optim is a work in progress. There are still some rough edges to be sanded down,
and features we want to implement. There are also planned breaking changes that
are good to be aware of. Please see the section on Planned Changes.


[github-img-url]: https://img.shields.io/badge/GitHub-source-green.svg
[github-repo-url]: https://github.com/JuliaNLSolvers/Optim.jl

[build-img]: https://travis-ci.org/JuliaNLSolvers/Optim.jl.svg?branch=master
[build-url]: https://travis-ci.org/JuliaNLSolvers/Optim.jl

[winbuild-img]: https://ci.appveyor.com/api/projects/status/prp8ygfp4rr9tafe?svg=true
[winbuild-url]: https://ci.appveyor.com/project/blegat/optim-jl

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
