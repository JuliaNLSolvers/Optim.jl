@testset "Literate examples" begin
    SKIPFILE = []

    EXAMPLEDIR = joinpath(@__DIR__, "../docs/src/examples")

    myfilter(str) = occursin(r"\.jl$", str) && !(str in SKIPFILE)
    for file in filter!(myfilter, readdir(EXAMPLEDIR))
        @testset "$file" begin
            mktempdir() do dir
                cd(dir) do
                    include(joinpath(EXAMPLEDIR, file))
                end
            end
        end
    end
end
