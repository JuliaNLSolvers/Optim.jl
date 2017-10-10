__precompile__(true)

module Optim
    using PositiveFactorizations
    using Compat
    using LineSearches
    using NLSolversBase
    using Calculus
#    using ReverseDiff
    using ForwardDiff

    import Compat.String
    import Compat.view

    import Base.length,
           Base.push!,
           Base.show,
           Base.getindex,
           Base.setindex!

    import NLSolversBase.iscomplex

    export optimize,
           NonDifferentiable,
           OnceDifferentiable,
           TwiceDifferentiable,
           OptimizationOptions,
           OptimizationState,
           OptimizationTrace,

           AcceleratedGradientDescent,
           BFGS,
           Brent,
           ConjugateGradient,
           Fminbox,
           GoldenSection,
           GradientDescent,
           LBFGS,
           MomentumGradientDescent,
           NelderMead,
           Newton,
           NewtonTrustRegion,
           SimulatedAnnealing,
           ParticleSwarm,

           Manifold,
           Flat,
           Sphere,
           Stiefel

    # Types
    include("types.jl")
    include("objective_types.jl")

    # Manifolds
    include("Manifolds.jl")

    # Generic stuff
    include("utilities/generic.jl")

    # Maxdiff
    include("utilities/maxdiff.jl")

    # Tracing
    include("utilities/update.jl")

    # Grid Search
    include("multivariate/solvers/zeroth_order/grid_search.jl")

    # Heuristic Optimization Methods
    include("multivariate/solvers/zeroth_order/nelder_mead.jl")
    include("multivariate/solvers/zeroth_order/simulated_annealing.jl")
    include("multivariate/solvers/zeroth_order/particle_swarm.jl")

    # preconditioning functionality
    include("multivariate/precon.jl")

    # Gradient Descent
    include("multivariate/solvers/first_order/gradient_descent.jl")
    include("multivariate/solvers/first_order/accelerated_gradient_descent.jl")
    include("multivariate/solvers/first_order/momentum_gradient_descent.jl")

    # Conjugate gradient
    include("multivariate/solvers/first_order/cg.jl")

    # (L-)BFGS
    include("multivariate/solvers/first_order/bfgs.jl")
    include("multivariate/solvers/first_order/l_bfgs.jl")

    # Newton
    include("multivariate/solvers/second_order/newton.jl")
    include("multivariate/solvers/second_order/newton_trust_region.jl")
    include("multivariate/solvers/second_order/krylov_trust_region.jl")

    # Constrained optimization
    include("multivariate/solvers/constrained/fminbox.jl")


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

    # Traces
    include("utilities/trace.jl")

    # API
    include("api.jl")

    # Examples for testing
    include(joinpath("problems", "unconstrained.jl"))
    include(joinpath("problems", "univariate.jl"))
end
