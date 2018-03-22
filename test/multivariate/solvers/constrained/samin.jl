@testset "SAMIN" begin
    MVP.UnconstrainedProblems.examples
    prob = probs["Himmelblau"]

    xtrue = prob.solutions
    f = OptimTestProblems.MultivariateProblems.objective(prob)
    x0 = prob.initial_x
    optimize(f, x0, x0.-100., x0.+100.0, Optim.SAMIN())
end
