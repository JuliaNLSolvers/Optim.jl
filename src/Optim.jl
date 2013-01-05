require("Options")
require("Distributions")
require("Calculus")

module Optim
    using OptionsMod
    using Distributions
    using Calculus

    function loadoptim(filename)
        include(joinpath(julia_pkgdir(), "Optim", "src", filename))
    end

    function centroid(p::Matrix)
         reshape(mean(p, 2), size(p, 1))
    end

    import Base.assign,
           Base.dot,
           Base.length,
           Base.push,
           Base.ref,
           Base.repl_show,
           Base.show

    export curve_fit,
           estimate_errors,
           optimize

    # Types
    loadoptim("types.jl")

    # RNG Sources
    loadoptim("rng.jl")

    # Grid Search
    loadoptim("grid_search.jl")

    # Line Search Methods
    loadoptim("backtracking_line_search.jl")

    # Gradient Descent Methods
    loadoptim("naive_gradient_descent.jl")
    loadoptim("gradient_descent.jl")

    # Conjugate gradient
    loadoptim("cgdescent.jl")

    # Newton and Quasi-Newton Methods
    loadoptim("newton.jl")
    loadoptim("bfgs.jl")
    loadoptim("l_bfgs.jl")

    # Constrained optimization
    loadoptim("fminbox.jl")

    # trust region methods
    loadoptim("levenberg_marquardt.jl")

    # Heuristic Optimization Methods
    loadoptim("nelder_mead.jl")
    loadoptim("simulated_annealing.jl")

    # End-User Facing Wrapper Functions
    loadoptim("optimize.jl")
    loadoptim("curve_fit.jl")
end
