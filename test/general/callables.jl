mutable struct MyCallable
    f
end
(a::MyCallable)(x) = a.f(x)
@testset "callables" begin
    @testset "univariate" begin
    # FIXME
    end
    @testset "multivariate" begin
        function rosenbrock(x::Vector)
           return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        end

        a = MyCallable(rosenbrock)

        optimize(a, rand(2), Optim.Options())
    end
end
