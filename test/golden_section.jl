f(x) = 2x^2+3x+1

results = optimize(f, -2.0, 1.0, method = GoldenSection())

@assert Optim.converged(results)
@assert abs(Optim.minimizer(results)+0.75) < 1e-7
