using Documenter, Optim

# use include("Rosenbrock.jl") etc

odir = Pkg.dir("Optim")
run(`cp $odir/LICENSE.md $odir/docs/src/LICENSE.md`)

#run('mv ../CONTRIBUTING.md ./dev/CONTRIBUTING.md') # TODO: Should we use the $odir/CONTRIBUTING.md file instead?
makedocs(
    doctest = false
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-windmill"),
    repo = "github.com/JuliaNLSolvers/Optim.jl.git",
    julia = "0.6"
)
