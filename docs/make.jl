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
    doctest = false,
    sitename = "Optim",
    pages = [
    "Home" => "index.md",
    "Tutorials" => [
        "Minimizing a function" => "user/minimization.md",
        "Gradients and Hessians" => "user/gradientsandhessians.md",
        "Configurable Options" => "user/config.md",
        "Linesearch" => "algo/linesearch.md",
        "Algorithm choice" => "user/algochoice.md",
        "Preconditioners" => "algo/precondition.md",
        "Complex optimization" => "algo/complex.md",
        "Manifolds" => "algo/manifolds.md",
        "Tips and tricks" => "user/tipsandtricks.md",
        "Interior point Newton" => "examples/generated/ipnewton_basics.md",
        "Maximum likelihood estimation" => "examples/generated/maxlikenlm.md",
        "Conditional maximum likelihood estimation" => "examples/generated/rasch.md",
        ],
    "Algorithms" => [
        "Gradient Free" => [
             "Nelder Mead" => "algo/nelder_mead.md",
             "Simulated Annealing" => "algo/simulated_annealing.md",
             "Simulated Annealing w/ bounds" => "algo/samin.md",
             "Particle Swarm" => "algo/particle_swarm.md",
             ],
        "Gradient Required" => [
             "Conjugate Gradient" => "algo/cg.md",
             "Gradient Descent" => "algo/gradientdescent.md",
             "(L-)BFGS" => "algo/lbfgs.md",
             "Acceleration" => "algo/ngmres.md",
             ],
        "Hessian Required" => [
             "Newton" => "algo/newton.md",
             "Newton with Trust Region" => "algo/newton_trust_region.md",
             "Interior point Newton" => "algo/ipnewton.md",
            ]
     ],
     "Contributing" => "dev/contributing.md",
     "License" => "LICENSE.md",
     ]
)

deploydocs(
    repo = "github.com/JuliaNLSolvers/Optim.jl.git",
)
