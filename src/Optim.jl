module Optim

  loadoptim(filename) = include(file_path(julia_pkgdir(),"Optim", "src", filename))

  export optimize, curve_fit, estimate_errors

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

  # Finite-Difference Methods
  loadoptim("estimate_gradient.jl")
  loadoptim("derivative.jl")
end
