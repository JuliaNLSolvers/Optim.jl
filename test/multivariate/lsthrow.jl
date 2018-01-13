@testset "Line search errors" begin
    function ls(df,x,s,xtmp,lsr,c,mayterminate)
        LineSearches._hagerzhang!(df,x,s,xtmp,lsr,c,mayterminate,
                                 0.1,0.9,
                                 Inf,5.,1e-6,0.66,
                                 2)
    end
    for optimizer in (ConjugateGradient, GradientDescent, LBFGS, BFGS, Newton, AcceleratedGradientDescent, MomentumGradientDescent)
        debug_printing && println("Testing $(string(optimizer))")
        prob = OptimTestProblems.UnconstrainedProblems.examples["Exponential"]
        @test_warn "Linesearch failed" optimize(UP.objective(prob), prob.initial_x,
                                                optimizer(alphaguess = LineSearches.InitialPrevious(),
                                                          linesearch = ls))
    end
end
