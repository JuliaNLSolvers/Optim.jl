@testset "Krylov Trust Region" begin

@testset "Toy test problem 1" begin
    # Test on actual optimization problems.
    srand(42)

    function f(x::Vector)
        (x[1] - 5.0)^4
    end

    function fg!(x::Vector, storage::Vector)
        storage[1] = 4.0 * (x[1] - 5.0)^3
        f(x)
    end

    function hv!(x::Vector, v::Vector, storage::Vector)
        storage[1] = 12.0 * (x[1] - 5.0)^2 * v[1]
    end

    d = TwiceDifferentiableHV(f, fg!, hv!)

    result = Optim.optimize(d, [0.0], KrylovTrustRegion())
    @test norm(Optim.minimizer(result) - [5.0]) < 0.01
end

@testset "Toy test problem 2" begin
    eta = 0.9

    function f2(x::Vector)
        0.5 * (x[1]^2 + eta * x[2]^2)
    end

    function fg2!(x::Vector, storage::Vector)
        storage[:] = [x[1], eta * x[2]]
        f2(x)
    end

    function hv2!(x::Vector, v::Vector, Hv::Vector)
        Hv[:] = [1.0 0.0; 0.0 eta] * v
    end

    d2 = TwiceDifferentiableHV(f2, fg2!, hv2!)

    result = Optim.optimize(d2, Float64[127, 921], KrylovTrustRegion())
    @test result.g_converged
    @test norm(Optim.minimizer(result) - [0.0, 0.0]) < 0.01
end

@testset "Stock test problems" begin
    for (name, prob) in Optim.UnconstrainedProblems.examples
    	if prob.istwicedifferentiable
            hv!(x::Vector, v::Vector, storage::Vector) = begin
                n = length(x)
                H = Matrix{Float64}(n, n)
                prob.h!(x, H)
                storage[:] = H * v
            end
            fg!(x::Vector, g::Vector) = begin
                prob.g!(x, g)
                prob.f(x)
            end
    		ddf = TwiceDifferentiableHV(prob.f, fg!, hv!)
    		result = Optim.optimize(ddf, prob.initial_x, KrylovTrustRegion())
    		@test norm(Optim.minimizer(result) - prob.solutions) < 1e-2
    	end
    end
end

end
