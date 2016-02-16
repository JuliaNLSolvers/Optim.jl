f_b(x) = 2x^2+3x+1

results = optimize(f_b, -2.0, 1.0, method = Brent())

@assert results.converged
@assert abs(results.minimum+0.75) < 1e-7
