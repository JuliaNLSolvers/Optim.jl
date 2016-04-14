p = Optim.UnconstrainedProblems.examples["Rosenbrock"]

df = DifferentiableFunction(p.f, p.g!)

Optim.optimize(df, [0.0, 0.0], method=MomentumGradientDescent())

Optim.optimize(df, [0.0, 0.0], method=MomentumGradientDescent(mu = 0.1))

optimize(p.f, p.g!, [0.0, 0.0])
