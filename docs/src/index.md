# Optim.jl
## How
Optim is registered in [METADATA.jl](https://github.com/JuliaLang/METADATA.jl).
This means that all you need to do to install Optim, is to run
```julia
Pkg.add("Optim")
```
## What
Optim is a Julia package for optimizing functions of
various kinds. While there is some support for box constrained optimization, most
of the solvers tries to find an $x$ that minimizes a function $f(x)$ without any constraints.
 Thus, the main focus is on unconstrained optimization.

## Why
There are many solvers available from both free and commercial sources, and many
of them are accessible from Julia. Few of them are written in Julia.
Performance-wise this is rarely a problem, as they are often written in either
Fortran or C. However, solvers written directly in Julia
does come with some advantages.

When writing Julia software (packages) that require something to be optimized, the programmer
can either choose to write their own optimization routine, or use one of the many
available solvers. For example, this could be something from the [NLOpt](...) suite.
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

## But...
Optim is a work in progress. There are still some rough edges to be sanded down,
and features we want to implement. There are also planned breaking changes that
are good to be aware of. Please see the section on Planned Changes.
