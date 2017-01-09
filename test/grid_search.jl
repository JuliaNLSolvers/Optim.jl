@testset "Grid Search" begin
    @test Optim.grid_search(x -> (1.0 - x)^2, [-1:0.1:1.0;]) == 1
end
# Cartesian product over 2 dimensions.
#grid_search(x -> (1.0 - x[1])^2 + (2.0 - x[2])^2, product[-1:0.1:1.0, -1:0.1:1.0])
