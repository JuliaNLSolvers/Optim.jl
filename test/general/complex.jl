# Regression tests for features that forgot about complex numbers

@testset "display complex result" begin
    grdt!(buf,u) = copyto!(buf, 2u)
    result = optimize(z->abs2.(z)[], grdt!, Complex.(randn(1)), ConjugateGradient())
    @test repr(result) isa String
end

@testset "default solver accepts complex" begin
    @test optimize(z->abs2.(z)[], Complex.(randn(1)))
end
