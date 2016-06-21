using Documenter, Optim

# use include("Rosenbrock.jl") etc

# assuming linux.
#run('mv ../LICENSE.md ./LICENSE.md')
#run('mv ../CONTRIBUTING.md ./dev/CONTRIBUTING.md')
makedocs(
    doctest = false
)

deploydocs(
    repo = "github.com/JuliaOpt/Optim.jl.git"
)
