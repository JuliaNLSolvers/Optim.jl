@testset "Simulated Annealing" begin
    Random.seed!(1)

    function f_s(x::Vector)
        (x[1] - 5.0)^4
    end
    options = Optim.Options(iterations=100_000)
    results = Optim.optimize(f_s, [0.0], SimulatedAnnealing(), options)
    @test norm(Optim.minimizer(results) - [5.0]) < 0.1

    function rosenbrock_s(x::Vector)
        (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    end
    options = Optim.Options(iterations=100_000)
    results = Optim.optimize(rosenbrock_s, [0.0, 0.0], SimulatedAnnealing(), options)
    @test norm(Optim.minimizer(results) - [1.0, 1.0]) < 0.1

    @suppress_out begin
        options = Optim.Options(iterations=10, show_trace=true, store_trace=true, extended_trace=true)
        results = Optim.optimize(rosenbrock_s, [0.0, 0.0], SimulatedAnnealing(), options)
    end
end
