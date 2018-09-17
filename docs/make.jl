if Base.HOME_PROJECT[] !== nothing
    # JuliaLang/julia/pull/28625
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

using Documenter, Optim

# use include("Rosenbrock.jl") etc
# Generate examples
include("generate.jl")

cp(joinpath(@__DIR__, "..", "LICENSE.md"),
   joinpath(@__DIR__, "src", "LICENSE.md"); force = true)

#run('mv ../CONTRIBUTING.md ./dev/CONTRIBUTING.md') # TODO: Should we use the $odir/CONTRIBUTING.md file instead?
makedocs(
    doctest = false
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-windmill"),
    repo = "github.com/JuliaNLSolvers/Optim.jl.git",
    julia = "1.0"
)
