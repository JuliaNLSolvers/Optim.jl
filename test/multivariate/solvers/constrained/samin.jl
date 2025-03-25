@testset "SAMIN" begin
    @testset "SAMIN $i" for i in keys(MVP.UnconstrainedProblems.examples)
        prob = MVP.UnconstrainedProblems.examples[i]
        if i !== "Himmelblau"
            continue
        end
        xtrue = prob.solutions
        f = OptimTestProblems.MultivariateProblems.objective(prob)
        x0 = prob.initial_x
        res = optimize(
            f,
            x0 ./ 1.1 .- 2.0,
            x0 .* 1.1 .+ 2.0,
            x0,
            Optim.SAMIN(t0 = 2.0, r_expand = 9.0, verbosity = 0),
            Optim.Options(iterations = 100000),
        )
        @test Optim.minimum(res) < 1e-5
    end
end
