@testset "optimize" begin
    # Tests for PR #302
    results = optimize(cos, 0, 2pi);
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0.0, 2pi);
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0, 2pi, Brent());
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0.0, 2pi, Brent());
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0, 2pi, method = Brent());
    @test norm(Optim.minimizer(results) - pi) < 0.01
    results = optimize(cos, 0.0, 2pi, method = Brent());
    @test norm(Optim.minimizer(results) - pi) < 0.01
end
