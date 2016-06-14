using Documenter, OptimDoc

# use include("Rosenbrock.jl") etc

# assuming linux.
#run('mv ../LICENSE.md ./LICENSE.md')
#run('mv ../CONTRIBUTING.md ./dev/CONTRIBUTING.md')
makedocs()

deploydocs(
    repo = "github.com/pkofod/OptimDoc.jl.git"
)
