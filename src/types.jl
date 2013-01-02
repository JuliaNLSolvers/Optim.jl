type OptimizationState
  state::Any
  f_state::Float64
  iteration::Int64
  metadata::Dict
end
function OptimizationState(s::Any, f::Float64, i::Int64)
  OptimizationState(s, f, i, Dict())
end

type OptimizationTrace
  states::Vector{OptimizationState}
end
OptimizationTrace() = OptimizationTrace(Array(OptimizationState, 0))

type OptimizationResults
  method::String
  initial_x::Vector{Float64}
  minimum::Vector{Float64}
  f_minimum::Float64
  iterations::Int64
  converged::Bool
  trace::OptimizationTrace
end

function show(io::IOStream, o_trace::OptimizationState)
  print(io, "State of Optimization Algorithm\n")
  print(io, " * Iteration: $(o_trace.iteration)\n")
  print(io, " * State: $(o_trace.state)\n")
  print(io, " * f(State): $(o_trace.f_state)\n")
  print(io, " * Additional Information: $(o_trace.metadata)")
end
repl_show(io::IOStream, s::OptimizationState) = show(io, s)
push(tr::OptimizationTrace, s::OptimizationState) = push(tr.states, s)
ref(tr::OptimizationTrace, i::Int64) = ref(tr.states, i)
assign(tr::OptimizationTrace, s::OptimizationState, i::Int64) = assign(tr.states, s, i)

function show(io::IOStream, results::OptimizationResults)
  print(io, "Results of Optimization Algorithm\n")
  print(io, " * Algorithm: $(results.method)\n")
  print(io, " * Starting Point: $(results.initial_x)\n")
  print(io, " * Minimum: $(results.minimum)\n")
  print(io, " * Value of Function at Minimum: $(results.f_minimum)\n")
  print(io, " * Iterations: $(results.iterations)\n")
  print(io, " * Self-Reported Convergence: $(results.converged)")
end
repl_show(io::IOStream, results::OptimizationResults) = show(io, results)
