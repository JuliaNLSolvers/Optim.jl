# Optim.jl

Univariate and multivariate optimization in Julia.

Optim.jl is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers) family.

| **Source**  | **Build Status** | **Social** | **References to cite** |
|:-:|:-:|:-:|:-:|
| [![Source](https://img.shields.io/badge/GitHub-source-green.svg)](https://github.com/JuliaNLSolvers/Optim.jl) |  [![Build Status](https://travis-ci.org/JuliaNLSolvers/Optim.jl.svg?branch=master)](https://travis-ci.org/JuliaNLSolvers/Optim.jl) | [![](https://badges.gitter.im/JuliaNLSolvers/Optim.jl.svg)](https://gitter.im/JuliaNLSolvers/Optim.jl) | [![JOSS](http://joss.theoj.org/papers/10.21105/joss.00615/status.svg)](https://doi.org/10.21105/joss.00615) |
| [![Codecov branch](https://img.shields.io/codecov/c/github/JuliaNLSolvers/Optim.jl/master.svg)](https://codecov.io/gh/JuliaNLSolvers/Optim.jl) |[![Build Status](https://ci.appveyor.com/api/projects/status/prp8ygfp4rr9tafe?svg=true)](https://ci.appveyor.com/project/blegat/optim-jl) |  | [![DOI](https://zenodo.org/badge/3933868.svg)](https://zenodo.org/badge/latestdoi/3933868) |


## What
Optim is a Julia package for optimizing functions of
various kinds. While there is some support for box constrained and Riemannian optimization, most
of the solvers try to find an ``x`` that minimizes a function ``f(x)`` without any constraints.
Thus, the main focus is on unconstrained optimization.
The provided solvers, under certain conditions, will converge to a local minimum.
In the case where a global minimum is desired we supply some methods such as (bounded) simulated annealing and particle swarm. For a dedicated package for global optimization techniques, see e.g. [BlackBoxOptim](https://github.com/robertfeldt/BlackBoxOptim.jl).

## Why
There are many solvers available from both free and commercial sources, and many
of them are accessible from Julia. Few of them are written in Julia.
Performance-wise this is rarely a problem, as they are often written in either
Fortran or C. However, solvers written directly in Julia
does come with some advantages.

When writing Julia software (packages) that require something to be optimized, the programmer
can either choose to write their own optimization routine, or use one of the many
available solvers. For example, this could be something from the [NLopt](https://github.com/JuliaOpt/NLopt.jl) suite.
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

The package is a registered package, and can be installed with `Pkg.add`.

```julia
julia> using Pkg; Pkg.add("Optim")
```
or through the `pkg` REPL mode by typing
```
] add Optim
```
