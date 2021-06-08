import Measurements
@testset "Measurements+Optim" begin
    #example problem in #823
    f(x)=(1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    xmes=zeros(Measurements.Measurement{Float64},2)
    xfloat = zeros(2)
    resmes = optimize(f,xmes)
    resfloat = optimize(f,xfloat)
    #given an initial value, they should give the exact same answer
    @test all(Optim.minimizer(resmes) .|> Measurements.value .== Optim.minimizer(resfloat))
    @test Optim.minimum(resmes) .|> Measurements.value .== Optim.minimum(resfloat)
    
end