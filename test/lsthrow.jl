@testset "Line search errors" begin
    function ls(df,x,s,xtmp,g,lsr,c,mayterminate)
        LineSearches.hagerzhang!(df,x,s,xtmp,g,lsr,c,mayterminate,
                                 0.1,0.9,
                                 Inf,5.,1e-6,0.66,
                                 2)
    end
    for optimizer in (ConjugateGradient, GradientDescent, LBFGS, BFGS, Newton, AcceleratedGradientDescent, MomentumGradientDescent)
        println("Testing $(string(optimizer))")
        prob = Optim.UnconstrainedProblems.examples["Exponential"]
        optimize(prob.f, prob.initial_x, optimizer(linesearch = ls))
        # TODO: How can I verify that the call to optimize gives a warning?
    end
end
