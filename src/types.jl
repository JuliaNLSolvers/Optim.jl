using Base

type OptimizationResults
  method::String
  initial_x::Vector
  minimum::Vector
  f_minimum::Float64
  iterations::Int64
  converged::Bool
end

function print(results::OptimizationResults)
  println()
  println("Results of Optimization Algorithm")
  println(" * Algorithm: $(results.method)")
  println(" * Starting Point: $(results.initial_x)")
  println(" * Minimum: $(results.minimum)")
  println(" * Value of Function at Minimum: $(results.f_minimum)")
  println(" * Iterations: $(results.iterations)")
  println(" * Self-Reported Convergence: $(results.converged)")
  println()
end

function show(results::OptimizationResults)
  print(results)
end
