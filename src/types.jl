type OptimizationResults
  method::String
  initial_x::Vector{Float64}
  minimum::Vector{Float64}
  f_minimum::Float64
  iterations::Int64
  converged::Bool
end

function show(io::IOStream, results::OptimizationResults)
  println(io)
  println(io, "Results of Optimization Algorithm")
  println(io, " * Algorithm: $(results.method)")
  println(io, " * Starting Point: $(results.initial_x)")
  println(io, " * Minimum: $(results.minimum)")
  println(io, " * Value of Function at Minimum: $(results.f_minimum)")
  println(io, " * Iterations: $(results.iterations)")
  println(io, " * Self-Reported Convergence: $(results.converged)")
  println(io)
end

repl_show(io::IOStream, results::OptimizationResults) = show(io, results)
