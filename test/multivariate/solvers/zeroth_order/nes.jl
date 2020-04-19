function rosenbrock2d(x::AbstractVector{T}) where T
    s=(1.0 - x[1])^2
    for i in 1:(length(x)-1)
        s+=100.0 * (x[i+1] - x[i]^2)^2
    end
    return s
end

@testset "sNES" begin
    f(x)=sum(abs2,x.-1.5)
    R=optimize(f,[1.0,2.0],sNES())
    @test f(R.minimizer)< 1e-5
    R=optimize(rosenbrock2d,[0.0,0.0],sNES(),Optim.Options(iterations=30000))
    @test rosenbrock2d(R.minimizer)< 1e-5
end

@testset "xNES" begin
    f(x)=sum(abs2,x.-1.5)
    R=optimize(f,[1.0,2.0],xNES())
    @test f(R.minimizer)< 1e-5
    R=optimize(rosenbrock2d,[0.0,0.0],xNES(),Optim.Options(iterations=30000))
    @test rosenbrock2d(R.minimizer)< 1e-5
end
