@testset "SAMIN" begin
    prob = MVP.UnconstrainedProblems.examples["Himmelblau"]

    xtrue = prob.solutions
    f = OptimTestProblems.MultivariateProblems.objective(prob)
    x0 = prob.initial_x
    optimize(f, x0, x0.-100., x0.+100.0, Optim.SAMIN(), Optim.Options(iterations=1000000))
end
