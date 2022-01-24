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
    for res in (
            Optim.optimize(f_prob, prob.initial_x, NelderMead()),
            Optim.optimize(f_prob, prob.initial_x, SimulatedAnnealing()),
            Optim.optimize(
                MVP.objective(prob),
                MVP.gradient(prob),
                prob.initial_x,
                LBFGS()
            )
        )
        @test typeof(f_prob(prob.initial_x)) == typeof(Optim.minimum(res))
        @test eltype(prob.initial_x) == eltype(Optim.minimizer(res))

        io = IOBuffer()
        show(io, res)
        s = String(take!(io))
        line_shift = res.method isa Union{SimulatedAnnealing,LBFGS} ? 5 : 1

        lines = split(s, '\n')
        @test lines[4] |> contains("Final objective value")
        @test lines[7] |> contains("Algorithm")
        @test lines[9] |> contains("Convergence measures")
        @test lines[13 + line_shift] |> contains("Iterations")
        @test lines[14 + line_shift] |> contains("f(x) calls")
        if res.method isa NelderMead
            @test lines[10] |> contains("√(Σ(yᵢ-ȳ)²)/n ≤ 1.0e-08")
        elseif res.method isa Union{SimulatedAnnealing,LBFGS}
            @test lines[10] |> contains("|x - x'|")
            @test lines[11] |> contains("|x - x'|/|x'|")
            @test lines[12] |> contains("|f(x) - f(x')|")
            @test lines[13] |> contains("|f(x) - f(x')|/|f(x')|")
            @test lines[14] |> contains("|g(x)|")
        end
    end

    io = IOBuffer()
    res = show(io, MIME"text/plain"(), Optim.Options(x_abstol = 10.0))
    @test String(take!(io)) |> contains("x_abstol = 10.0")
end
