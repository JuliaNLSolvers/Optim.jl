module TestTypes
    using Base.Test
    using Compat
    using Optim

    prob = Optim.UnconstrainedProblems.examples["Rosenbrock"]
    f_prob = prob.f
    res = Optim.optimize(f_prob, prob.initial_x, method=NelderMead())

    io = IOBuffer()
    show(io, res)
    s = takebuf_string(io)

    lines = split(s, '\n')
    @test lines[1] == "Results of Optimization Algorithm"
    @test startswith(lines[2], " * Algorithm: ")
    @test startswith(lines[3], " * Starting Point: ")
    @test startswith(lines[4], " * Minimizer: [")
    @test startswith(lines[5], " * Minimum: ")
    @test startswith(lines[6], " * Iterations: ")
    @test startswith(lines[7], " * Convergence: ")
    @test startswith(lines[8], "   * |x - x'| < ")
    @test startswith(lines[9], "   * |f(x) - f(x')| / |f(x)| < ")
    @test startswith(lines[10], "   * |g(x)| < ")
    @test startswith(lines[11], "   * Reached Maximum Number of Iterations: ")
    @test startswith(lines[12], " * Objective Function Calls: ")
    @test startswith(lines[13], " * Gradient Calls: ")
end
