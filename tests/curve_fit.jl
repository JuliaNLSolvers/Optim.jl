require("Optim")
using Optim

# fitting noisy data to an exponential model
model(xpts, p) = p[1]*exp(-xpts.*p[2])

# some example data
srand(12345)
xpts = linspace(0,10,20)
data = model(xpts, [1.0 2.0]) + 0.01*randn(length(xpts))

beta, r, J = curve_fit(model, xpts, data, [0.5, 0.5])
println("Found beta: $beta")
@assert norm(beta - [1.0, 2.0]) < 0.05

# can also get error estimates on the fit parameters
errors = estimate_errors(beta, r, J)

println("Estimated errors: $errors")