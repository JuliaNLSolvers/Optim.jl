@testset "successive_f_tol" begin
    alg = GradientDescent(
        alphaguess = LineSearches.InitialStatic(),
        linesearch = LineSearches.Static(),
    )
    opt = Optim.Options(
        iterations = 10,
        successive_f_tol = 5,
        f_tol = 3,
        g_tol = -1,
    )
    result = Optim.optimize(
        sum,
        (y, _) -> fill!(y, 1),
        [0.0, 0.0],
        alg,
        opt,
    )
    @test result.iterations == opt.successive_f_tol + 1
end
