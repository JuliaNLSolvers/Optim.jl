__precompile__(true)

module Optim
    using Calculus
    using PositiveFactorizations
    using Compat
    using ForwardDiff
    using LineSearches

    import Compat.String
    import Compat.view

    import Base.length,
           Base.push!,
           Base.show,
           Base.getindex,
           Base.setindex!

    export optimize,
           DifferentiableFunction,
           TwiceDifferentiableFunction,
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

    # API
    include("api.jl")

    # Generic stuff
    include("utilities/generic.jl")

    # Maxdiff
    include("utilities/maxdiff.jl")

    # Tracing
    include("utilities/update.jl")

    # Line search generic code
    include("utilities/perform_linesearch.jl")

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

    # trust region methods
    include("levenberg_marquardt.jl")

    # Heuristic Optimization Methods
    include("nelder_mead.jl")
    include("simulated_annealing.jl")
    include("particle_swarm.jl")

    # Univariate methods
    include("golden_section.jl")
    include("brent.jl")

    # Backward compatibility
    include("deprecate.jl")

    # End-User Facing Wrapper Functions
    include("optimize.jl")

    # Convergence
    include("utilities/assess_convergence.jl")

    # Traces
    include("utilities/trace.jl")

    # Examples for testing
    include(joinpath("problems", "unconstrained.jl"))
    include(joinpath("problems", "univariate.jl"))
end
