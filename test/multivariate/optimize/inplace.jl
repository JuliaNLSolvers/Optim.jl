@testset "inplace keyword" begin
    rosenbrock = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
    f = MVP.objective(rosenbrock)
    g! = MVP.gradient(rosenbrock)
    h! = MVP.hessian(rosenbrock)
    function g(x)
        G = similar(x)
        g!(G, x)
        G
    end
    function h(x)
        n = length(x)
        H = similar(x, n, n)
        h!(H, x)
        H
    end
    initial_x = rosenbrock.initial_x

    inp_res = optimize(f, g, h, initial_x; inplace = false)
    op_res  = optimize(f, g!, h!, initial_x; inplace = true)

    for op in (Optim.minimizer, Optim.minimum, Optim.f_calls,
                Optim.g_calls, Optim.h_calls, Optim.iterations, Optim.converged)
        @test all(op(inp_res) .=== op(op_res))
    end
end
