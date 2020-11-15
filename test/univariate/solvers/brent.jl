@testset "Brent's Method" begin
    for (name, prob) in OptimTestProblems.UnivariateProblems.examples
        for T in (Float64, BigFloat)
            results = optimize(prob.f, convert(Array{T}, prob.bounds)..., method = Brent())

            @test Optim.converged(results)
            @test norm(Optim.minimizer(results) .- prob.minimizers) < 1e-7
        end
    end

    ## corner cases - empty and zero-width brackets
    result = optimize(sqrt, 4.0, 4.0, method = Brent())
    @test Optim.converged(result)
    @test Optim.minimizer(result) == 4.0
    @test Optim.minimum(result) == 2.0
    @test_throws ErrorException optimize(identity, 2.0, 1.0, Brent())
    @test summary(result) == "Brent's Method"

    ## corner cases - largely flat functions
    result = optimize(x->sign(x), -2, 2)
    @test Optim.converged(result)
    @test Optim.minimum(result) == -1.0
    result = optimize(x->sign(x), -1, 2)
    @test Optim.converged(result)
    @test Optim.minimum(result) == -1.0
    result = optimize(x->sign(x), -2, 1)
    @test Optim.converged(result)
    @test Optim.minimum(result) == -1.0

    result = Optim.optimize(x->sin(x), 0, 2π, Optim.Brent(); abs_tol=1e-4, store_trace=false, show_trace=true, iterations=2)
end
