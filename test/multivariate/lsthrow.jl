@testset "Line search errors" begin
    hz = LineSearches.HagerZhang(; delta = 0.1, sigma = 0.9, alphamax = Inf,
                      rho = 5.0, epsilon = 1e-6, gamma = 0.66, linesearchmax = 2)
    for optimizer in (ConjugateGradient, GradientDescent, LBFGS, BFGS, Newton, AcceleratedGradientDescent, MomentumGradientDescent)
        debug_printing && println("Testing $(string(optimizer))")
        prob = MultivariateProblems.UnconstrainedProblems.examples["Exponential"]
        @test optimize(MVP.objective(prob), prob.initial_x, optimizer(alphaguess = LineSearches.InitialPrevious(), linesearch = hz)).ls_success == false
    end
end
