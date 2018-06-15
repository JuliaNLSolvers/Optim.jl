@testset "Newton" begin
    function f_1(x::Vector)
        (x[1] - 5.0)^4
    end

    function g!_1(storage::Vector, x::Vector)
        storage[1] = 4.0 * (x[1] - 5.0)^3
    end

    function h!_1(storage::Matrix, x::Vector)
        storage[1, 1] = 12.0 * (x[1] - 5.0)^2
    end
    initial_x = [0.0]

    Optim.optimize(NonDifferentiable(f_1, initial_x), [0.0], Newton())
    Optim.optimize(OnceDifferentiable(f_1, g!_1, initial_x), [0.0], Newton())

    options = Optim.Options(store_trace = false, show_trace = false,
                            extended_trace = true)
    results = Optim.optimize(f_1, g!_1, h!_1, [0.0], Newton(), options)
    @test_throws ErrorException Optim.x_trace(results)
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [5.0]) < 0.01

    eta = 0.9

    function f_2(x::Vector)
      (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    function g!_2(storage::Vector, x::Vector)
      storage[1] = x[1]
      storage[2] = eta * x[2]
    end

    function h!_2(storage::Matrix, x::Vector)
      storage[1, 1] = 1.0
      storage[1, 2] = 0.0
      storage[2, 1] = 0.0
      storage[2, 2] = eta
    end

    results = Optim.optimize(f_2, g!_2, h!_2, [127.0, 921.0], Newton())
    @test_throws ErrorException Optim.x_trace(results)
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01
    @test summary(results) == "Newton's Method"

    @testset "newton in concave region" begin
        prob=MultivariateProblems.UnconstrainedProblems.examples["Himmelblau"]
        res = optimize(MVP.objective(prob), MVP.gradient(prob), MVP.hessian(prob), [0., 0.], Newton())
        @test norm(Optim.minimizer(res) - prob.solutions) < 1e-9
    end

    @testset "Optim problems" begin
        run_optim_tests(Newton(); skip = ("Trigonometric",), show_name = debug_printing)
    end
end
