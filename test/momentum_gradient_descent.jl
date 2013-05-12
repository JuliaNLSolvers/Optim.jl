p = Optim.UnconstrainedProblems.examples["Rosenbrock"]

df = DifferentiableFunction(p.f, p.g!)

Optim.momentum_gradient_descent(df, [0.0, 0.0])

Optim.momentum_gradient_descent(df, [0.0, 0.0], mu = 0.1)

optimize(p.f, p.g!, [0.0, 0.0])
