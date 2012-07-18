load("src/init.jl")

# Define a simple function.
function f(x::Vector)
  (100.0 - x[1])^2 + (50.0 - x[2])^2
end

# Check that gradient is approximately accurate.
g = estimate_gradient(f)
@assert norm(g([100.0, 50.0]) - [0.0, 0.0]) < 0.01

# Run optimize() using approximate gradient.
results = optimize(f, g, [0.0, 0.0])
@assert norm(results.minimum - [100.0, 50.0]) < 0.01

# Compare with running optimize() without using finite differencing.
results = optimize(f, [0.0, 0.0])
@assert norm(results.minimum - [100.0, 50.0]) < 0.01
