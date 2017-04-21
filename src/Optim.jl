__precompile__(true)

module Optim
    using PositiveFactorizations
    using Compat
    using LineSearches
    using NLSolversBase
    using Calculus
    using ReverseDiff
    using ForwardDiff

    import Compat.String
    import Compat.view

    import Base.length,
           Base.push!,
           Base.show,
           Base.getindex,
           Base.setindex!

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
           ParticleSwarm

    # Types
    include("types.jl")
    include("objective_types.jl")

    # Generic stuff
    include("utilities/generic.jl")

    # Maxdiff
    include("utilities/maxdiff.jl")

    # Tracing
    include("utilities/update.jl")

    # Grid Search
    include("grid_search.jl")

    # preconditioning functionality
    include("precon.jl")

    # Gradient Descent
    include("gradient_descent.jl")
    include("accelerated_gradient_descent.jl")
    include("momentum_gradient_descent.jl")

    # Conjugate gradient
    include("cg.jl")

    # Newton and Quasi-Newton Methods
    include("newton.jl")
    include("newton_trust_region.jl")
    include("bfgs.jl")
    include("l_bfgs.jl")

    # Constrained optimization
    include("fminbox.jl")

    # Heuristic Optimization Methods
    include("nelder_mead.jl")
    include("simulated_annealing.jl")
    include("particle_swarm.jl")

    # Univariate methods
    include("univariate/golden_section.jl")
    include("univariate/brent.jl")
    include("univariate/types.jl")
    include("univariate/printing.jl")

    # Line search generic code
    include("utilities/perform_linesearch.jl")

    # Backward compatibility
    include("deprecate.jl")

    # convenient user facing optimize methods
    include("optimize/univariate/interface.jl")
    include("optimize/multivariate/interface.jl")

    # actual optimize methods
    include("optimize/univariate/optimize.jl")
    include("optimize/multivariate/optimize.jl")

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
