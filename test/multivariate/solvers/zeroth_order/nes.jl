@testset "sNES" begin
    f(x)=sum(abs2,x.-1.5)
    R=optimize(f,[1.0,2.0],sNES())
    @test f(R.minimizer)< 1e-5
end
