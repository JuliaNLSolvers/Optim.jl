"""
# Optim.jl
Welcome to Optim.jl!

Optim.jl is a package used to solve continuous optimization problems. It is
written in Julia for Julians to help take advantage of arbitrary number types,
fast computation, and excellent automatic differentiation tools.

## REPL help
`?` followed by an algorithm name (`?BFGS`), constructors (`?Optim.Options`)
prints help to the terminal.

## Documentation
Besides the help provided at the REPL, it is possible to find help and general
documentation online at http://julianlsolvers.github.io/Optim.jl/stable/ .
"""
module Optim
using NLSolversBase          # for shared infrastructure in JuliaNLSolvers
using PositiveFactorizations # for globalization strategy in Newton
import PositiveFactorizations: cholesky!, cholesky
using LineSearches           # for globalization strategy in Quasi-Newton algs
using DiffEqDiffTools        # for finite difference derivatives
# using ReverseDiff           # for reverse mode AD (not really suitable for scalar output)
using ForwardDiff            # for forward mode AD
import NaNMath               # for functions that ignore NaNs (no poisoning)

import Parameters: @with_kw, # for types where constructors are simply defined
                   @unpack   # by their default values, and simple unpacking
                             # of fields

using Printf                 # For printing, maybe look into other options

using FillArrays             # For handling scalar bounds in Fminbox

#using Compat                 # for compatibility across multiple julia versions

# for extensions of functions defined in Base.
import Base: length, push!, show, getindex, setindex!, maximum, minimum

# objective and constraints types and functions relevant to them.
import NLSolversBase: NonDifferentiable, OnceDifferentiable, TwiceDifferentiable,
                      nconstraints, nconstraints_x, NotInplaceObjective, InplaceObjective

# var for NelderMead
import StatsBase: var

import LinearAlgebra: Diagonal, diag, Hermitian, Symmetric,
                      rmul!, mul!, ldiv!,
                      dot, norm, normalize!,
                      eigen, BLAS,
                      cholesky, Cholesky, # factorizations
                      I,
                      svd
import SparseArrays: AbstractSparseMatrix

# exported functions and types
export optimize, maximize, # main function

       # Re-export objective types from NLSolversBase
       NonDifferentiable,
       OnceDifferentiable,
       TwiceDifferentiable,

       # Re-export constraint types from NLSolversBase
       TwiceDifferentiableConstraints,

       # I don't think these should be here [pkofod]
       OptimizationState,
       OptimizationTrace,

       # Optimization algorithms
       ## Zeroth order methods (heuristics)
       NelderMead,
       ParticleSwarm,
       SimulatedAnnealing,

       ## First order
       ### Quasi-Newton
       GradientDescent,
       BFGS,
       LBFGS,

       ### Conjugate gradient
       ConjugateGradient,

       ### Acceleration methods
       AcceleratedGradientDescent,
       MomentumGradientDescent,

       ### Nonlinear GMRES
       NGMRES,
       OACCEL,

       ## Second order
       ### (Quasi-)Newton
       Newton,

       ### Trust region
       NewtonTrustRegion,

       # Constrained
       ## Box constraints, x_i in [lb_i, ub_i]
       ### Specifically Univariate, R -> R
       GoldenSection,
       Brent,

       ### Multivariate, R^N -> R
       Fminbox,
       SAMIN,

       ## Manifold constraints
       Manifold,
       Flat,
       Sphere,
       Stiefel,

       ## Non-linear constraints
       IPNewton


include("types.jl") # types used throughout
include("Manifolds.jl") # code to handle manifold constraints
include("multivariate/precon.jl") # preconditioning functionality

# utilities
include("utilities/generic.jl") # generic utilities
include("utilities/maxdiff.jl") # find largest difference
include("utilities/update.jl")  # trace code

# Unconstrained optimization
## Grid Search
include("multivariate/solvers/zeroth_order/grid_search.jl")

## Zeroth order (Heuristic) Optimization Methods
include("multivariate/solvers/zeroth_order/nelder_mead.jl")
include("multivariate/solvers/zeroth_order/simulated_annealing.jl")
include("multivariate/solvers/zeroth_order/particle_swarm.jl")

## Quasi-Newton
include("multivariate/solvers/first_order/gradient_descent.jl")
include("multivariate/solvers/first_order/bfgs.jl")
include("multivariate/solvers/first_order/l_bfgs.jl")

## Acceleration methods
include("multivariate/solvers/first_order/accelerated_gradient_descent.jl")
include("multivariate/solvers/first_order/momentum_gradient_descent.jl")

## Conjugate gradient
include("multivariate/solvers/first_order/cg.jl")


## Newton
### Line search
include("multivariate/solvers/second_order/newton.jl")
include("multivariate/solvers/second_order/krylov_trust_region.jl")
### Trust region
include("multivariate/solvers/second_order/newton_trust_region.jl")

## Nonlinear GMRES
include("multivariate/solvers/first_order/ngmres.jl")

# Constrained optimization
## Box constraints
include("multivariate/solvers/constrained/fminbox.jl")
include("multivariate/solvers/constrained/samin.jl")

# Univariate methods
include("univariate/solvers/golden_section.jl")
include("univariate/solvers/brent.jl")
include("univariate/types.jl")
include("univariate/printing.jl")

# Line search generic code
include("utilities/perform_linesearch.jl")

# Backward compatibility
include("deprecate.jl")

# convenient user facing optimize methods
include("univariate/optimize/interface.jl")
include("multivariate/optimize/interface.jl")

# actual optimize methods
include("univariate/optimize/optimize.jl")
include("multivariate/optimize/optimize.jl")

# Convergence
include("utilities/assess_convergence.jl")
include("multivariate/solvers/zeroth_order/zeroth_utils.jl")

# Traces
include("utilities/trace.jl")

# API
include("api.jl")

## Interior point includes
include("multivariate/solvers/constrained/ipnewton/types.jl")
# Tracing
include("multivariate/solvers/constrained/ipnewton/utilities/update.jl")
# Constrained optimization
include("multivariate/solvers/constrained/ipnewton/iplinesearch.jl")
include("multivariate/solvers/constrained/ipnewton/interior.jl")
include("multivariate/solvers/constrained/ipnewton/ipnewton.jl")
# Convergence
include("multivariate/solvers/constrained/ipnewton/utilities/assess_convergence.jl")
# Traces
include("multivariate/solvers/constrained/ipnewton/utilities/trace.jl")

# Maximization convenience wrapper
include("maximize.jl")

end
