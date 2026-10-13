@testset "BFGS" begin
    # Trigonometric ends up in a local minimum
    skip = ("Trigonometric",)
    run_optim_tests(
        BFGS();
        skip,
        show_name = debug_printing,
    )

    @testset "Symmetric inverse Hessian approximation" begin
        options = Optim.Options(store_trace = true, extended_trace = true)

        prob = MVP.UnconstrainedProblems.examples["Extended Rosenbrock"]
        res = optimize(
            MVP.objective(prob),
            MVP.gradient(prob),
            prob.initial_x,
            BFGS(),
            options,
        )
        @test Optim.converged(res)
        @test all(s -> issymmetric(s.metadata["~inv(H)"]), Optim.trace(res))

        Random.seed!(0)
        n = 4
        A = randn(ComplexF64, n, n)
        A = A'A + I
        b = randn(ComplexF64, n)
        fcomplex(x) = real(dot(x, A * x) / 2 - dot(b, x)) + sum(abs.(x) .^ 4)
        gcomplex!(g, x) = copyto!(g, A * x - b + 4 * (abs.(x) .^ 2) .* x)
        res = optimize(fcomplex, gcomplex!, randn(ComplexF64, n), BFGS(), options)
        @test Optim.converged(res)
        @test all(s -> ishermitian(s.metadata["~inv(H)"]), Optim.trace(res))
    end
end
