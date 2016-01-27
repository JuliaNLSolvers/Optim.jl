isdefined(Base, :__precompile__) && __precompile__()

module Optim
    using Calculus, PositiveFactorizations
    using Compat

    import Base.length,
           Base.push!,
           Base.show,
           Base.getindex,
           Base.setindex!

    export optimize,
           interior,
           linlsq,
           DifferentiableFunction,
           TwiceDifferentiableFunction,
           ConstraintsBox

    # Types
    include("types.jl")

    # Types for constrained optimization
    include("constraints.jl")

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
    include("nnls.jl")

    # trust region methods
    include("levenberg_marquardt.jl")

    # Heuristic Optimization Methods
    include("nelder_mead.jl")
    include("simulated_annealing.jl")

    # Univariate methods
    include("golden_section.jl")
    include("brent.jl")

    # Constrained optimization algorithms
    include("interior.jl")

    # End-User Facing Wrapper Functions
    include("optimize.jl")

    cgdescent(args...) = error("API has changed. Please use cg.")

    # Tests
    const basedir = dirname(Base.source_path())
    const testpaths = [joinpath(basedir, "problems", "unconstrained.jl"),
                       joinpath(basedir, "problems", "constrained.jl")]

end
