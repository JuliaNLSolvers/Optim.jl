@testset "default solvers" begin
    prob = MVP.UnconstrainedProblems.examples["Powell"]
    f = objective(prob)
    g! = gradient(prob)
    h! = hessian(prob)
    @test summary(optimize(f, prob.initial_x)) == summary(NelderMead())
    @test summary(optimize(f, g!, prob.initial_x)) == summary(LBFGS())
    @test summary(optimize(f, g!, h!, prob.initial_x)) == summary(Newton())
end
