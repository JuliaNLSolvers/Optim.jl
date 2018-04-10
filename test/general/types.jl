using Compat
import Compat.String

@testset "Types" begin
    solver = NelderMead()
    T = typeof(solver)
    trace = OptimizationTrace{Float64, T}()
    push!(trace,OptimizationState{Float64, T}(1,1.0,1.0,Dict()))
    push!(trace,OptimizationState{Float64, T}(2,1.0,1.0,Dict()))
    @test length(trace) == 2
    @test trace[end].iteration == 2

    prob = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
    f_prob = MVP.objective(prob)
    for g_free in (NelderMead(), SimulatedAnnealing())
        res = Optim.optimize(f_prob, prob.initial_x, g_free)
        @test typeof(f_prob(prob.initial_x)) == typeof(Optim.minimum(res))
        @test eltype(prob.initial_x) == eltype(Optim.minimizer(res))

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
            @test startswith(lines[10], " * Objective Calls: ")
        elseif res.method == "Simulated Annealing"
            @test startswith(lines[8], "   * |x - x'| ≤ ")
            @test startswith(lines[9], "   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|")
            @test startswith(lines[10], "   * |g(x)| ≤ ")
            @test startswith(lines[11], "   * Stopped by an increasing objective:")
            @test startswith(lines[12], "   * Reached Maximum Number of Iterations: ")
            @test startswith(lines[13], " * Objective Calls: ")
        end
    end

    res = Optim.optimize(MVP.objective(prob), MVP.gradient(prob), prob.initial_x, LBFGS())
    @test typeof(f_prob(prob.initial_x)) == typeof(Optim.minimum(res))
    @test eltype(prob.initial_x) == eltype(Optim.minimizer(res))


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
    @test startswith(lines[8], "   * |x - x'| ≤ ")
    @test startswith(lines[9],  "     |x - x'| = ")
    @test startswith(lines[10], "   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|:") # TODO: regexp
    @test startswith(lines[11], "     |f(x) - f(x')| = 7.43e+10 |f(x)|") # TODO: regexp
    @test startswith(lines[12], "   * |g(x)| ≤ ")
    @test startswith(lines[13], "     |g(x)| = ")
    @test startswith(lines[14], "   * Stopped by an increasing objective:")
    @test startswith(lines[15], "   * Reached Maximum Number of Iterations: ")
    @test startswith(lines[16], " * Objective Calls: ")
    @test startswith(lines[17], " * Gradient Calls: ")
    if res.method in ("Newton's Method", "Newton's Method (Trust Region)")
        @test startswith(lines[18], " * Hessian Calls: ")
    end
end
