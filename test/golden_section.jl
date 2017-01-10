@testset "Golden Section" begin
    for (name, prob) in Optim.UnivariateProblems.examples
        for T in (Float64, BigFloat)
            results = optimize(prob.f, convert(Array{T}, prob.bounds)...,
                               method = GoldenSection())

            @test Optim.converged(results)
            @test norm(Optim.minimizer(results) - prob.minimizers) < 1e-7
        end
    end
end
