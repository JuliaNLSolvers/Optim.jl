module Optim
  using Base

  export optimize, curve_fit, estimate_errors

  # Types
  include(find_in_path("Optim/src/types.jl"))

  # RNG Sources
  include(find_in_path("Optim/src/rng.jl"))

  # Grid Search
  include(find_in_path("Optim/src/grid_search.jl"))

  # Line Search Methods
  include(find_in_path("Optim/src/backtracking_line_search.jl"))

  # Gradient Descent Methods
  include(find_in_path("Optim/src/naive_gradient_descent.jl"))
  include(find_in_path("Optim/src/gradient_descent.jl"))

  # Newton and Quasi-Newton Methods
  include(find_in_path("Optim/src/newton.jl"))
  include(find_in_path("Optim/src/bfgs.jl"))
  include(find_in_path("Optim/src/l_bfgs.jl"))

  # trust region methods
  include(find_in_path("Optim/src/levenberg_marquardt.jl"))

  # Heuristic Optimization Methods
  include(find_in_path("Optim/src/nelder_mead.jl"))
  include(find_in_path("Optim/src/simulated_annealing.jl"))

  # End-User Facing Wrapper Functions
  include(find_in_path("Optim/src/optimize.jl"))
  include(find_in_path("Optim/src/curve_fit.jl"))

  # Finite-Difference Methods
  include(find_in_path("Optim/src/estimate_gradient.jl"))
  include(find_in_path("Optim/src/derivative.jl"))
end
