VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module Optim
    using Calculus
    using Compat

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
           SimulatedAnnealing

    # Types
    include("types.jl")

    # API
    include("api.jl")

    # Automatic differentiation utilities
    include("autodiff.jl")

    # Maxdiff
    include("utilities/maxdiff.jl")

    # Tracing
    include("utilities/update.jl")

    # Convergence
    include("utilities/assess_convergence.jl")

    # Grid Search
    include("grid_search.jl")

    # Line Search Methods
    include(joinpath("linesearch", "backtracking_linesearch.jl"))
    include(joinpath("linesearch", "interpolating_linesearch.jl"))
    include(joinpath("linesearch", "mt_cstep.jl"))
    include(joinpath("linesearch", "mt_linesearch.jl"))
    include(joinpath("linesearch", "hz_linesearch.jl"))

    # Gradient Descent
    include("gradient_descent.jl")
    include("accelerated_gradient_descent.jl")
    include("momentum_gradient_descent.jl")

    # Conjugate gradient
    include("cg.jl")

    # Newton and Quasi-Newton Methods
    include("newton.jl")
    include("bfgs.jl")
    include("l_bfgs.jl")

    # Constrained optimization
    include("fminbox.jl")

    # trust region methods
    include("levenberg_marquardt.jl")

    # Heuristic Optimization Methods
    include("nelder_mead.jl")
    include("simulated_annealing.jl")

    # Univariate methods
    include("golden_section.jl")
    include("brent.jl")

    # Backward compatibility
    include("deprecate.jl")

    # End-User Facing Wrapper Functions
    include("optimize.jl")

    # Examples for testing
    include(joinpath("problems", "unconstrained.jl"))

    cgdescent(args...) = error("API has changed. Please use cg.")
end
