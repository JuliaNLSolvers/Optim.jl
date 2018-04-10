using Documenter, Optim

# use include("Rosenbrock.jl") etc

# assuming linux.
run('mv ../LICENSE.md ./LICENSE.md')
#run('mv ../CONTRIBUTING.md ./dev/CONTRIBUTING.md')
makedocs(
    doctest = false
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    repo = "github.com/JuliaNLSolvers/Optim.jl.git",
    julia = "0.6"
)
