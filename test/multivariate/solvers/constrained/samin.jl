using Random
Random.seed!(3288)
@testset "SAMIN" begin
    @testset "SAMIN $i" for i in keys(MVP.UnconstrainedProblems.examples)
        prob = MVP.UnconstrainedProblems.examples[i]
        xtrue = prob.solutions
        if length(xtrue) > 10 || i == "Powell" || i == "Parabola"
            continue
        end
        f = OptimTestProblems.MultivariateProblems.objective(prob)
        x0 = prob.initial_x
        res = optimize(
            f,
            xtrue ./ 1.1,
            xtrue .* 1.1,
            xtrue .* 1.02,
            Optim.SAMIN(t0 = 1.0, r_expand = 2.0, verbosity = 0),
            Optim.Options(iterations = 10000),
        )
        @test abs(prob.minimum-Optim.minimum(res)) < 1e-5
    end

    @testset "Normal convergence returns results" begin
        Random.seed!(1234)
        res = optimize(
            x -> x[1]^2,
            [-1.0],
            [1.0],
            [0.5],
            SAMIN(),
            Optim.Options(iterations = 100_000, x_abstol = 1e-3, f_abstol = 1e-12),
        )
        @test Optim.minimizer(res) ≈ [0.0] atol = 1e-3
        @test Optim.minimum(res) < 1e-12
        @test Optim.x_converged(res)
        @test Optim.f_converged(res)
        @test !Optim.iteration_limit_reached(res)
    end
end
