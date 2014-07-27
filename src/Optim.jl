require("Options")

module Optim
    using OptionsMod
    using Distributions
    using Calculus

    import Base.dot,
           Base.length,
           Base.push!,
           Base.show,
           Base.getindex,
           Base.setindex!

    export curve_fit,
           estimate_errors,
           optimize,
           minfinder,
           DifferentiableFunction,
           TwiceDifferentiableFunction

    # Types
    include("types.jl")

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
    include("cgdescent.jl")
    include("cg.jl")

    # Newton and Quasi-Newton Methods
    include("newton.jl")
    include("bfgs.jl")
    include("l_bfgs.jl")

    # Constrained optimization
    include("fminbox.jl")
    include("nnls.jl")

    # Multiple minima
    include("minfinder.jl")

    # trust region methods
    include("levenberg_marquardt.jl")

    # Heuristic Optimization Methods
    include("nelder_mead.jl")
    include("simulated_annealing.jl")

    # Univariate methods
    include("golden_section.jl")
    include("brent.jl")

    # End-User Facing Wrapper Functions
    include("optimize.jl")
    include("curve_fit.jl")

    # Examples for testing
    include(joinpath("problems", "unconstrained.jl"))
    include(joinpath("problems", "multiple_minima.jl"))
end
