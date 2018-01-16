@testset "Golden Section" begin
    for (name, prob) in OptimTestProblems.UnivariateProblems.examples
        for T in (Float64, BigFloat)
            results = optimize(prob.f, convert(Array{T}, prob.bounds)...,
                               method = GoldenSection())

            @test Optim.converged(results)
            @test norm(Optim.minimizer(results) .- prob.minimizers) < 1e-7
        end
    end

    ## corner cases - empty and zero-width brackets
    result = optimize(sqrt, 4.0, 4.0, method = GoldenSection())
    @test Optim.converged(result)
    @test Optim.minimizer(result) == 4.0
    @test Optim.minimum(result) == 2.0
    @test_throws ErrorException optimize(identity, 2.0, 1.0, GoldenSection())

end
