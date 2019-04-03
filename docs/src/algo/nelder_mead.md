# Nelder-Mead
Nelder-Mead is currently the standard algorithm when no derivatives are provided.
## Constructor
```julia
NelderMead(; parameters = AdaptiveParameters(),
             initial_simplex = AffineSimplexer())
```
The keywords in the constructor are used to control the following parts of the
solver:

* `parameters` is a an instance of either `AdaptiveParameters` or `FixedParameters`, and is
used to generate parameters for the Nelder-Mead Algorithm.
* `initial_simplex` is an instance of `AffineSimplexer`. See more
details below.


## Description
Our current implementation of the Nelder-Mead algorithm is based on Nelder and Mead (1965) and
Gao and Han (2010). Gradient free methods can be a bit sensitive to starting values
and tuning parameters, so it is a good idea to be careful with the defaults provided
in Optim.

Instead of using gradient information, Nelder-Mead is a direct search method.
It keeps track of the function value at a number
of points in the search space. Together, the points form a simplex. Given a simplex,
we can perform one of four actions: reflect, expand, contract, or shrink. Basically,
the goal is to iteratively replace the worst point with a better point. More information
can be found in Nelder and Mead (1965), Lagarias, et al (1998) or Gao and Han (2010).

The stopping rule is the same as in the original paper, and is the standard
error of the function values at the vertices. To set the tolerance level for this
convergence criterion, set the `g_tol` level as described in the Configurable Options
section.

When the solver finishes, we return a minimizer which is either the centroid or one of the vertices.
The function value at the centroid adds a function evaluation, as we need to evaluate the objection
at the centroid to choose the smallest function value. However, even if the function value at the centroid can be returned
as the minimum, we do not trace it during the optimization iterations. This is to avoid
too many evaluations of the objective function which can be computationally expensive.
Typically, there should be no more than twice as many `f_calls` than `iterations`.
 Adding an evaluation at the centroid when tracing could considerably increase the total
run-time of the algorithm.

### Specifying the initial simplex
The default choice of `initial_simplex` is `AffineSimplexer()`. A simplex is represented
by an ``(n+1)``-dimensional vector of ``n``-dimensional vectors. It is used together
 with the initial `x` to create the initial simplex. To
construct the ``i``th vertex, it simply multiplies entry ``i`` in the initial vector with
a constant `b`, and adds a constant `a`. This means that the ``i``th of the ``n`` additional
vertices is of the form

```math
(x_0^1, x_0^2, \ldots, x_0^i, \ldots, 0,0) + (0, 0, \ldots, x_0^i\cdot b+a,\ldots, 0,0)
```

If an ``x_0^i`` is zero, we need the ``a`` to make sure all vertices are unique. Generally,
it is advised to start with a relatively large simplex.

If a specific simplex is wanted, it is possible to construct the ``(n+1)``-vector of ``n``-dimensional vectors,
and pass it to the solver using a new type definition and a new method for the function `simplexer`.
For example, let us minimize the two-dimensional Rosenbrock function, and choose three vertices that have elements
that are simply standard uniform draws.
```julia
using Optim
struct MySimplexer <: Optim.Simplexer end
Optim.simplexer(S::MySimplexer, initial_x) = [rand(length(initial_x)) for i = 1:length(initial_x)+1]
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
optimize(f, [.0, .0], NelderMead(initial_simplex = MySimplexer()))
```

Say we want to implement the initial simplex as in Matlab's `fminsearch`. This is very close
to the `AffineSimplexer` above, but with a small twist. Instead of always adding the `a`,
a constant is only added to entries that are zero. If the entry is non-zero, five
percent of the level is added. This might be implemented (by the user) as
```julia
struct MatlabSimplexer{T} <: Optim.Simplexer
    a::T
    b::T
end
MatlabSimplexer(;a = 0.00025, b = 0.05) = MatlabSimplexer(a, b)

function Optim.simplexer(A::MatlabSimplexer, initial_x::AbstractArray{T, N}) where {T, N}
    n = length(initial_x)
    initial_simplex = Array{T, N}[initial_x for i = 1:n+1]
    for j = 1:n
        initial_simplex[j+1][j] += initial_simplex[j+1][j] == zero(T) ? S.b * initial_simplex[j+1][j] : S.a
    end
    initial_simplex
end
```

### The parameters of Nelder-Mead
The different types of steps in the algorithm are governed by four parameters:
``\alpha`` for the reflection, ``\beta`` for the expansion, ``\gamma`` for the contraction,
and ``\delta`` for the shrink step. We default to the adaptive parameters scheme in
Gao and Han (2010). These are based on the dimensionality of the problem, and
are given by

```math
\alpha = 1, \quad \beta = 1+2/n,\quad \gamma =0.75 - 1/2n,\quad \delta = 1-1/n
```

It is also possible to specify the original parameters from Nelder and Mead (1965)

```math
\alpha = 1,\quad \beta = 2, \quad\gamma = 1/2, \quad\delta = 1/2
```

by specifying `parameters  = Optim.FixedParameters()`. For specifying custom values,
`parameters  = Optim.FixedParameters(α = a, β = b, γ = g, δ = d)` is used, where a, b, g, d are the chosen values. If another
parameter specification is wanted, it is possible to create a custom sub-type of`Optim.NMParameters`,
and add a method to the `parameters` function. It should take the new type as the
first positional argument, and the dimensionality of `x` as the second positional argument, and
return a 4-tuple of parameters. However, it will often be easier to simply supply
the wanted parameters to `FixedParameters`.
## References
Nelder, John A. and R. Mead (1965). "A simplex method for function minimization". Computer Journal 7: 308–313. doi:10.1093/comjnl/7.4.308.

Lagarias, Jeffrey C., et al. "Convergence properties of the Nelder--Mead simplex method in low dimensions." SIAM Journal on optimization 9.1 (1998): 112-147.

Gao, Fuchang and Lixing Han (2010). "Implementing the Nelder-Mead simplex algorithm with adaptive parameters". Computational Optimization and Applications [DOI 10.1007/s10589-010-9329-3]
