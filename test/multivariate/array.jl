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
        for m in (AcceleratedGradientDescent, ConjugateGradient, BFGS, LBFGS, NelderMead, GradientDescent, MomentumGradientDescent, NelderMead, ParticleSwarm, SimulatedAnnealing, NGMRES, OACCEL)
            debug_printing && printstyled("Solver: "*string(m); color = :green)
            res = optimize(f, g!, [1., 0., 1., 0.], m())
            @test typeof(Optim.minimizer(res)) <: Vector
            if !(m in (NelderMead, SimulatedAnnealing, ParticleSwarm))
                @test norm(Optim.minimizer(res) - [10.0, 0.0, 0.0, 5.0]) < 10e-8
            end
        end
    end

    @testset "matrix" begin
        for m in (AcceleratedGradientDescent, ConjugateGradient, BFGS, LBFGS, ConjugateGradient,  GradientDescent, MomentumGradientDescent, ParticleSwarm, SimulatedAnnealing, NGMRES, OACCEL)
            res = optimize(f, g!, Matrix{Float64}(I, 2, 2), m())
            @test typeof(Optim.minimizer(res)) <: Matrix
            if !(m in (SimulatedAnnealing, ParticleSwarm))
                @test norm(Optim.minimizer(res) - [10.0 0.0; 0.0 5.0]) < 10e-8
            end
        end
    end

    @testset "tensor" begin
        eye3 = zeros(2,2,1)
        eye3[:,:,1] = Matrix{Float64}(I, 2, 2)
        for m in (AcceleratedGradientDescent, ConjugateGradient, BFGS, LBFGS, ConjugateGradient,  GradientDescent, MomentumGradientDescent, ParticleSwarm, SimulatedAnnealing, NGMRES, OACCEL)
            res = optimize(f, g!, eye3, m())
            _minimizer = Optim.minimizer(res)
            @test typeof(_minimizer) <: Array{Float64, 3}
            @test size(_minimizer) == (2,2,1)
            if !(m in (SimulatedAnnealing, ParticleSwarm))
                @test norm(_minimizer - [10.0 0.0; 0.0 5.0]) < 10e-8
            end
        end
    end
end

using RecursiveArrayTools
@testset "arraypartition input" begin

    function polynomial(x)
            return (10.0 - x[1])^2 + (7.0 - x[2])^4 + (108.0 - x[3])^4
        end

    function polynomial_gradient!(storage, x)
            storage[1] = -2.0 * (10.0 - x[1])
            storage[2] = -4.0 * (7.0 - x[2])^3
            storage[3] = -4.0 * (108.0 - x[3])^3
        end

    function polynomial_hessian!(storage, x)
            storage[1, 1] = 2.0
            storage[1, 2] = 0.0
            storage[1, 3] = 0.0
            storage[2, 1] = 0.0
            storage[2, 2] = 12.0 * (7.0 - x[2])^2
            storage[2, 3] = 0.0
            storage[3, 1] = 0.0
            storage[3, 2] = 0.0
            storage[3, 3] = 12.0 * (108.0 - x[3])^2
        end

    ap = ArrayPartition(rand(1), rand(2))

    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, NelderMead())
    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, ParticleSwarm())
    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, SimulatedAnnealing())

    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, GradientDescent())
    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, AcceleratedGradientDescent())
    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, MomentumGradientDescent())

    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, ConjugateGradient())

    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, BFGS())
    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, LBFGS())

    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, Newton())
    optimize(polynomial, polynomial_gradient!, polynomial_hessian!, ap, NewtonTrustRegion())
end
