using Documenter, Optim

makedocs(
    doctest = false
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    repo = "github.com/JuliaNLSolvers/Optim.jl.git",
    julia = "0.6"
)
