@testset "package issues" begin
    @test isempty(detect_ambiguities(Optim))
end
