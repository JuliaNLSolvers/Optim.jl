type MyCallables
    f
end
@testset "callables" begin
    (a::MyCallables)(x) = a.f(x)

    function rosenbrock(x::Vector)
       return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    end

    a = MyCallables(rosenbrock)

    optimize(a, rand(2), Optim.Options())
end
