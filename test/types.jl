using Compat
import Compat.String

@testset "Types" begin
    solver = NelderMead()
    T = typeof(solver)
    trace = OptimizationTrace{T}()
    push!(trace,OptimizationState{T}(1,1.0,1.0,Dict()))
    push!(trace,OptimizationState{T}(2,1.0,1.0,Dict()))
    @test length(trace) == 2
    @test trace[end].iteration == 2

    prob = Optim.UnconstrainedProblems.examples["Rosenbrock"]
    f_prob = prob.f
    for g_free in (NelderMead(), SimulatedAnnealing())
        res = Optim.optimize(f_prob, prob.initial_x, g_free)

        io = IOBuffer()
        show(io, res)
        s = String(take!(io))

        lines = split(s, '\n')
        @test lines[1] == "Results of Optimization Algorithm"
        @test startswith(lines[2], " * Algorithm: ")
        @test startswith(lines[3], " * Starting Point: ")
        @test startswith(lines[4], " * Minimizer: [")
        @test startswith(lines[5], " * Minimum: ")
        @test startswith(lines[6], " * Iterations: ")
        @test startswith(lines[7], " * Convergence: ")
        if res.method == "Nelder-Mead"
            @test startswith(lines[8], "   *  √(Σ(yᵢ-ȳ)²)/n < 1.0e-08: ")
            @test startswith(lines[9], "   * Reached Maximum Number of Iterations: ")
            @test startswith(lines[10], " * Objective Function Calls: ")
        elseif res.method == "Simulated Annealing"
            @test startswith(lines[8], "   * |x - x'| < ")
            @test startswith(lines[9], "   * |f(x) - f(x')| / |f(x)| < ")
            @test startswith(lines[10], "   * |g(x)| < ")
            @test startswith(lines[12], "   * Reached Maximum Number of Iterations: ")
            @test startswith(lines[13], " * Objective Function Calls: ")
        end
    end

    res = Optim.optimize(prob.f, prob.g!, prob.initial_x, LBFGS())

    io = IOBuffer()
    show(io, res)
    s = String(take!(io))

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
    @test startswith(lines[11], "   * f(x) > f(x')")
    @test startswith(lines[12], "   * Reached Maximum Number of Iterations: ")
    @test startswith(lines[13], " * Objective Function Calls: ")
    @test startswith(lines[14], " * Gradient Calls: ")
end
