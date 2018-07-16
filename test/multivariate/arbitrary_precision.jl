# TODO: Add HigherPrecision.jl tests here?
# TODO: Test Interior Point algorithm as well
@testset "Arbitrary Precision" begin
    prob = MVP.UnconstrainedProblems.examples["Rosenbrock"]

    f = MVP.objective(prob)
    g! = MVP.gradient(prob)
    h! = MVP.hessian(prob)

    @testset "BigFloat" begin
        x0 = big.(prob.initial_x)
        res = optimize(f, x0)
        debug_printing && @show res
        @test Optim.converged(res) == true
        @test Optim.minimum(res) < 1e-8
        @test Optim.minimizer(res) ≈ [1.0, 1.0] atol=1e-4 rtol=0


        res = optimize(f, g!, x0)
        debug_printing && @show res
        @test Optim.converged(res) == true
        @test Optim.minimum(res) < 1e-16
        @test Optim.minimizer(res) ≈ [1.0, 1.0] atol=1e-10 rtol=0

        res = optimize(f, g!, h!, x0)
        debug_printing && @show res
        @test Optim.converged(res) == true
        @test Optim.minimum(res) < 1e-16
        @test Optim.minimizer(res) ≈ [1.0, 1.0] atol=1e-10 rtol=0

        lower = big.([-Inf, -Inf])
        upper = big.([0.5, 1.5])
        res = optimize(f, g!, lower, upper, x0)
        debug_printing && @show res
        @test Optim.converged(res) == true
        @test Optim.minimum(res) ≈ 0.25 atol=1e-10 rtol=0
        @test Optim.minimizer(res) ≈ [0.5, 0.25] atol=1e-10 rtol=0

        res = optimize(f, lower, upper, x0)
        debug_printing && @show res
        @test Optim.converged(res) == true
        @test Optim.minimum(res) ≈ 0.25 atol=1e-10 rtol=0
        @test Optim.minimizer(res) ≈ [0.5, 0.25] atol=1e-10 rtol=0
    end
end
