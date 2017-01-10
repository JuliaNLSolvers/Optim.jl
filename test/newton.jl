let
    function f_1(x::Vector)
        (x[1] - 5.0)^4
    end

    function g!_1(x::Vector, storage::Vector)
        storage[1] = 4.0 * (x[1] - 5.0)^3
    end

    function h!_1(x::Vector, storage::Matrix)
        storage[1, 1] = 12.0 * (x[1] - 5.0)^2
    end

    d = TwiceDifferentiableFunction(f_1, g!_1, h!_1)

    # Need to specify autodiff!
    @test_throws ErrorException Optim.optimize(DifferentiableFunction(f_1, g!_1), [0.0], Newton())
    Optim.optimize(DifferentiableFunction(f_1, g!_1), [0.0], Newton(), Optim.Options(autodiff = true))

    results = Optim.optimize(d, [0.0], Newton())
    @test_throws ErrorException Optim.x_trace(results)
    @assert Optim.g_converged(results)
    @assert norm(Optim.minimizer(results) - [5.0]) < 0.01

    eta = 0.9

    function f_2(x::Vector)
      (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    function g!_2(x::Vector, storage::Vector)
      storage[1] = x[1]
      storage[2] = eta * x[2]
    end

    function h!_2(x::Vector, storage::Matrix)
      storage[1, 1] = 1.0
      storage[1, 2] = 0.0
      storage[2, 1] = 0.0
      storage[2, 2] = eta
    end

    d = TwiceDifferentiableFunction(f_2, g!_2, h!_2)
    results = Optim.optimize(d, [127.0, 921.0], Newton())
    @test_throws ErrorException Optim.x_trace(results)
    @assert Optim.g_converged(results)
    @assert norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

    # Test Optim.newton for all twice differentiable functions in Optim.UnconstrainedProblems.examples
    for (name, prob) in Optim.UnconstrainedProblems.examples
    	if prob.istwicedifferentiable
    		ddf = TwiceDifferentiableFunction(prob.f, prob.g!,prob.h!)
    		res = Optim.optimize(ddf, prob.initial_x, Newton())
    		@assert norm(Optim.minimizer(res) - prob.solutions) < 1e-2
    	end
    end

    let
        prob=Optim.UnconstrainedProblems.examples["Himmelblau"]
        ddf = TwiceDifferentiableFunction(prob.f, prob.g!, prob.h!)
        res = optimize(ddf, [0., 0.], Newton())
        @assert norm(Optim.minimizer(res) - prob.solutions) < 1e-9
    end


    for (name, prob) in Optim.UnconstrainedProblems.examples
    	if prob.istwicedifferentiable
    		ddf = DifferentiableFunction(prob.f, prob.g!)
    		res = Optim.optimize(ddf, prob.initial_x, Newton(), Optim.Options(autodiff = true))
    		@assert norm(Optim.minimizer(res) - prob.solutions) < 1e-2
    		res = Optim.optimize(ddf.f, prob.initial_x, Newton(), Optim.Options(autodiff = true))
    		@assert norm(Optim.minimizer(res) - prob.solutions) < 1e-2
            res = Optim.optimize(ddf.f, ddf.g!, prob.initial_x, Newton(), Optim.Options(autodiff = true))
    		@assert norm(Optim.minimizer(res) - prob.solutions) < 1e-2
    	end
    end
end
