using Optim, Compat
if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end
debug_printing = false

my_tests = [
    "types.jl",
    "bfgs.jl",
    "gradient_descent.jl",
    "accelerated_gradient_descent.jl",
    "momentum_gradient_descent.jl",
    "grid_search.jl",
    "l_bfgs.jl",
    "newton.jl",
    "newton_trust_region.jl",
    "cg.jl",
    "nelder_mead.jl",
    "optimize.jl",
    "simulated_annealing.jl",
    "particle_swarm.jl",
    "golden_section.jl",
    "brent.jl",
    "type_stability.jl",
    "array.jl",
    "constrained.jl",
    "callbacks.jl",
    "precon.jl",
    "initial_convergence.jl",
    "extrapolate.jl",
    "levenberg_marquardt.jl",
    "lsthrow.jl",
    "api.jl",
]

for my_test in my_tests
    include(my_test)
end
