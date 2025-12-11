@testset "Krylov Trust Region" begin

    @testset "Toy test problem 1" begin
        # Test on actual optimization problems.
        function f(x::Vector)
            (x[1] - 5.0)^4
        end
        function fg!(_f, g, x)
            if g !== nothing
                g[1] = 4.0 * (x[1] - 5.0)^3
            end
            return _f === nothing ? nothing : f(x)
        end

        function hv!(Hv, x, v)
            Hv[1] = 12.0 * (x[1] - 5.0)^2 * v[1]
        end
        d = TwiceDifferentiable(NLSolversBase.only_fg_and_hv!(fg!, hv!), [0.0])
        result = Optim.optimize(d, [0.0], Optim.KrylovTrustRegion())
        @test norm(Optim.minimizer(result) - [5.0]) < 0.01

        function fgh!(f, g, H, x)
            if H !== nothing
                H[1, 1] = 12.0 * (x[1] - 5.0)^2
            end
            return fg!(f, g, x)
        end
        d2 = TwiceDifferentiable(NLSolversBase.only_fgh!(fgh!), [0.0])
        result = Optim.optimize(d2, [0.0], Optim.KrylovTrustRegion())
        @test norm(Optim.minimizer(result) - [5.0]) < 0.01
    end

    @testset "Toy test problem 2" begin
        eta = 0.9

        function f2(x::Vector)
            0.5 * (x[1]^2 + eta * x[2]^2)
        end

        function fg2!(_f, _g, x::Vector)
            if _g !== nothing
                _g[1] = x[1]
                _g[2] = eta * x[2]
            end
            return _f === nothing ? nothing : f2(x)
        end

        function hv2!(Hv::Vector, x::Vector, v::Vector)
            return mul!(Hv, Diagonal([1.0, eta]), v)
        end

        d2 = Optim.TwiceDifferentiable(NLSolversBase.only_fg_and_hv!(fg2!, hv2!), Float64[127, 921])

        result = Optim.optimize(d2, Float64[127, 921], Optim.KrylovTrustRegion())
        @test Optim.g_converged(result)
        @test norm(Optim.minimizer(result) - [0.0, 0.0]) < 0.01
    end

    @testset "Stock test problems" begin
        for (name, prob) in MultivariateProblems.UnconstrainedProblems.examples
            if prob.istwicedifferentiable
                n = length(prob.initial_x)
                hv! = let H = Matrix{Float64}(undef, n, n)
                    function (storage::Vector, x::Vector, v::Vector)
                        MVP.hessian(prob)(H, x)
                        return mul!(storage, H, v)
                    end
                end
                fg!(_f, _g, x::Vector) = begin
                    if _g !== nothing
                        MVP.gradient(prob)(_g, x)
                    end
                    return _f === nothing ? nothing : MVP.objective(prob)(x)
                end
                ddf = Optim.TwiceDifferentiable(
                    NLSolversBase.only_fg_and_hv!(fg!, hv!),
                    prob.initial_x,
                )
                result = Optim.optimize(ddf, prob.initial_x, Optim.KrylovTrustRegion())
                @test norm(Optim.minimizer(result) - prob.solutions) < 1e-2
            end
        end
    end

end
