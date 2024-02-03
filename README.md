# Optim.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianlsolvers.github.io/Optim.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://julianlsolvers.github.io/Optim.jl/dev)
[![Build Status](https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/windows.yml/badge.svg)](https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/windows.yml)
[![Build Status](https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/linux.yml/badge.svg)](https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/linux.yml)
[![Build Status](https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/mac.yml/badge.svg)](https://github.com/JuliaNLSolvers/Optim.jl/actions/workflows/mac.yml)
[![Codecov branch](https://img.shields.io/codecov/c/github/JuliaNLSolvers/Optim.jl/master.svg)](https://codecov.io/gh/JuliaNLSolvers/Optim.jl)
[![JOSS](http://joss.theoj.org/papers/10.21105/joss.00615/status.svg)](https://doi.org/10.21105/joss.00615)

Univariate and multivariate optimization in Julia.

Optim.jl is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers)
family.

## Help and support

For help and support, please post on the [Optimization (Mathematical)](https://discourse.julialang.org/c/domain/opt/13)
section of the Julia discourse or the `#optimization` channel of the Julia [slack](https://julialang.org/slack/).

## Installation

Install `Optim.jl` using the Julia package manager:
```julia
import Pkg
Pkg.add("Optim")
```

## Documentation

The online documentation is available at [https://julianlsolvers.github.io/Optim.jl/stable](https://julianlsolvers.github.io/Optim.jl/stable).

## Example

To minimize the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function),
do:
```julia
julia> using Optim

julia> rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
rosenbrock (generic function with 1 method)

julia> result = optimize(rosenbrock, zeros(2), BFGS())
 * Status: success

 * Candidate solution
    Final objective value:     5.471433e-17

 * Found with
    Algorithm:     BFGS

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

julia> Optim.minimizer(result)
2-element Vector{Float64}:
 0.9999999926033423
 0.9999999852005355

julia> Optim.minimum(result)
5.471432670590216e-17
```

To get information on the keywords used to construct method instances, use the
Julia REPL help prompt (`?`)
```julia
help?> LBFGS
search: LBFGS

  LBFGS
  ≡≡≡≡≡

  Constructor
  ===========

  LBFGS(; m::Integer = 10,
  alphaguess = LineSearches.InitialStatic(),
  linesearch = LineSearches.HagerZhang(),
  P=nothing,
  precondprep = (P, x) -> nothing,
  manifold = Flat(),
  scaleinvH0::Bool = true && (typeof(P) <: Nothing))

  LBFGS has two special keywords; the memory length m, and the scaleinvH0 flag.
  The memory length determines how many previous Hessian approximations to
  store. When scaleinvH0 == true, then the initial guess in the two-loop
  recursion to approximate the inverse Hessian is the scaled identity, as can be
  found in Nocedal and Wright (2nd edition) (sec. 7.2).

  In addition, LBFGS supports preconditioning via the P and precondprep keywords.

  Description
  ===========

  The LBFGS method implements the limited-memory BFGS algorithm as described in
  Nocedal and Wright (sec. 7.2, 2006) and original paper by Liu & Nocedal
  (1989). It is a quasi-Newton method that updates an approximation to the
  Hessian using past approximations as well as the gradient.

  References
  ==========

    •  Wright, S. J. and J. Nocedal (2006), Numerical optimization, 2nd edition.
       Springer

    •  Liu, D. C. and Nocedal, J. (1989). "On the Limited Memory Method for
       Large Scale Optimization". Mathematical Programming B. 45 (3): 503–528
```

## Use with JuMP

You can use Optim.jl with [JuMP.jl](https://github.com/jump-dev/JuMP.jl) as
follows:

```julia
julia> using JuMP, Optim

julia> model = Model(Optim.Optimizer);

julia> set_optimizer_attribute(model, "method", BFGS())

julia> @variable(model, x[1:2]);

julia> @objective(model, Min, (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2)
(x[1]² - 2 x[1] + 1) + (100.0 * ((-x[1]² + x[2]) ^ 2.0))

julia> optimize!(model)

julia> objective_value(model)
3.7218241804173566e-21

julia> value.(x)
2-element Vector{Float64}:
 0.9999999999373603
 0.99999999986862
```

## Citation

If you use `Optim.jl` in your work, please cite the following:

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
