@testset "Dual numbers" begin
  f(x,y) = x[1]^2+y[1]^2
  g(param) = Optim.minimum(optimize(x->f(x,param), -1,1))
  @test Optim.NLSolversBase.ForwardDiff.gradient(g, [1.0]) == [2.0]
end
