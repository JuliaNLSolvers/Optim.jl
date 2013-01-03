# Define a simple function.
function f(x::Vector)
  (100.0 - x[1])^2 + (50.0 - x[2])^2
end

#
# Forward differencing
#

# Check that gradient is approximately accurate.
g = estimate_gradient(f)
@assert norm(g([100.0, 50.0]) - [0.0, 0.0]) < 0.01

# Run optimize() using approximate gradient.
results = optimize(f, g, [0.0, 0.0])
@assert norm(results.minimum - [100.0, 50.0]) < 0.01

#
# Central differencing
#

# Check that gradient is approximately accurate.
g = estimate_gradient2(f)
@assert norm(g([100.0, 50.0]) - [0.0, 0.0]) < 0.01

# Run optimize() using approximate gradient.
results = optimize(f, g, [0.0, 0.0])
@assert norm(results.minimum - [100.0, 50.0]) < 0.01

# Comparison

# Compare both results with running optimize() without using any form of finite differencing.
results = optimize(f, [0.0, 0.0])
@assert norm(results.minimum - [100.0, 50.0]) < 0.01
