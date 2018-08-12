@testset "Krylov Trust Region" begin

@testset "Toy test problem 1" begin
    # Test on actual optimization problems.
    function f(x::Vector)
        (x[1] - 5.0)^4
    end

    function fg!(storage::Vector, x::Vector)
        storage[1] = 4.0 * (x[1] - 5.0)^3
        f(x)
    end

    function hv!(storage::Vector, x::Vector, v::Vector)
        storage[1] = 12.0 * (x[1] - 5.0)^2 * v[1]
    end

    d = Optim.TwiceDifferentiableHV(f, fg!, hv!, [0.0])

    result = Optim.optimize(d, [0.0], Optim.KrylovTrustRegion())
    @test norm(Optim.minimizer(result) - [5.0]) < 0.01
end

@testset "Toy test problem 2" begin
    eta = 0.9

    function f2(x::Vector)
        0.5 * (x[1]^2 + eta * x[2]^2)
    end

    function fg2!(storage::Vector, x::Vector)
        storage[:] = [x[1], eta * x[2]]
        f2(x)
    end

    function hv2!(Hv::Vector, x::Vector, v::Vector)
        Hv[:] = [1.0 0.0; 0.0 eta] * v
    end

    d2 = Optim.TwiceDifferentiableHV(f2, fg2!, hv2!, Float64[127, 921])

    result = Optim.optimize(d2, Float64[127, 921], Optim.KrylovTrustRegion())
    @test result.g_converged
    @test norm(Optim.minimizer(result) - [0.0, 0.0]) < 0.01
end

@testset "Stock test problems" begin
    for (name, prob) in MultivariateProblems.UnconstrainedProblems.examples
      if prob.istwicedifferentiable
            hv!(storage::Vector, x::Vector, v::Vector) = begin
                n = length(x)
                H = Matrix{Float64}(undef, n, n)
                MVP.hessian(prob)(H, x)
                storage .= H * v
            end
            fg!(g::Vector, x::Vector) = begin
                MVP.gradient(prob)(g,x)
                MVP.objective(prob)(x)
            end
        ddf = Optim.TwiceDifferentiableHV(MVP.objective(prob), fg!, hv!, prob.initial_x)
        result = Optim.optimize(ddf, prob.initial_x, Optim.KrylovTrustRegion())
        @test norm(Optim.minimizer(result) - prob.solutions) < 1e-2
      end
    end
end

end
