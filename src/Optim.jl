module Optim
  using Base

  export optimize

  # Types
  load("Optim/src/types.jl")

  # RNG Sources
  load("Optim/src/rng.jl")

  # Grid Search
  load("Optim/src/grid_search.jl")

  # Line Search Methods
  load("Optim/src/backtracking_line_search.jl")

  # Gradient Descent Methods
  load("Optim/src/naive_gradient_descent.jl")
  load("Optim/src/gradient_descent.jl")

  # Newton and Quasi-Newton Methods
  load("Optim/src/newton.jl")
  load("Optim/src/bfgs.jl")
  load("Optim/src/l_bfgs.jl")

  # Heuristic Optimization Methods
  load("Optim/src/nelder_mead.jl")
  load("Optim/src/simulated_annealing.jl")

  # End-User Facing Wrapper Functions
  load("Optim/src/optimize.jl")

  # Finite-Difference Methods
  load("Optim/src/estimate_gradient.jl")
  load("Optim/src/derivative.jl")
end
