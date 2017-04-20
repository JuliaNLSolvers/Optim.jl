@testset "input types" begin
    f(X) = (10 - X[1])^2 + (0 - X[2])^2 + (0 - X[3])^2 + (5 - X[4])^2

    function g!(storage, x)
        storage[1] = -20 + 2 * x[1]
        storage[2] = 2 * x[2]
        storage[3] = 2 * x[3]
        storage[4] = -10 + 2 * x[4]
        return
    end
    @testset "vector" begin
        for m in (AcceleratedGradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), NelderMead(), GradientDescent(), MomentumGradientDescent(), NelderMead(), SimulatedAnnealing(), ParticleSwarm())
            res = optimize(f, g!, [1., 0., 1., 0.], m)

            @test typeof(Optim.minimizer(res)) <: Vector
            @test vecnorm(Optim.minimizer(res) - [10.0, 0.0, 0.0, 5.0]) < 10e-8
        end
    end

    # PSO does not accept matrix input
    @testset "matrix" begin
        for m in (AcceleratedGradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), NelderMead(), GradientDescent(), MomentumGradientDescent(), NelderMead(), SimulatedAnnealing())
            res = optimize(f, g!, eye(2), m)

            @test typeof(Optim.minimizer(res)) <: Matrix
            @test vecnorm(Optim.minimizer(res) - [10.0 0.0; 0.0 5.0]) < 10e-8
        end
    end

    # PSO does not accept tensor input
    @testset "tensor" begin
        eye3 = zeros(2,2,1)
        eye3[:,:,1] = eye(2)
        for m in (AcceleratedGradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), NelderMead(), GradientDescent(), MomentumGradientDescent(), NelderMead(), SimulatedAnnealing())
            res = optimize(f, g!, eye3, m)
            _minimizer = Optim.minimizer(res)
            @test typeof(_minimizer) <: Array{Float64, 3}
            @test size(_minimizer) == (2,2,1)
            @test vecnorm(_minimizer - [10.0 0.0; 0.0 5.0]) < 10e-8
        end
    end
end
