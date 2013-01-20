#
# Correctness Tests
#

using Optim

my_tests = [#"test/bfgs.jl", # TODO: Make this pass
            "test/curve_fit.jl",
            "test/gradient_descent.jl",
            "test/grid_search.jl",
            "test/l_bfgs.jl",
            "test/levenberg_marquardt.jl",
            "test/naive_gradient_descent.jl",
            "test/newton.jl",
            "test/nelder_mead.jl",
            "test/optimize.jl",
            "test/simulated_annealing.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
