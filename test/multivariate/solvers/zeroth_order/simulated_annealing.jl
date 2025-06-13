@testset "Simulated Annealing" begin
    Random.seed!(1)

    function f_s(x::Vector)
        (x[1] - 5.0)^4
    end
    options = Optim.Options(iterations = 100_000)
    results = Optim.optimize(f_s, [0.0], SimulatedAnnealing(), options)
    @test norm(Optim.minimizer(results) - [5.0]) < 0.1

    function rosenbrock_s(x::Vector)
        (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    end
    options = Optim.Options(iterations = 100_000)
    results = Optim.optimize(rosenbrock_s, [0.0, 0.0], SimulatedAnnealing(), options)
    @test norm(Optim.minimizer(results) - [1.0, 1.0]) < 0.1

    options = Optim.Options(
        iterations = 10,
        show_trace = true,
        store_trace = true,
        extended_trace = true,
    )
    results = Optim.optimize(rosenbrock_s, [0.0, 0.0], SimulatedAnnealing(), options)

    # Max-cut problem, https://en.wikipedia.org/wiki/Maximum_cut
    maxcut_objective(x::AbstractVector, J::AbstractMatrix{Bool}) = x' * (J * x)
    function maxcut_spinflip!(xcurrent::AbstractVector, xproposed::AbstractVector, p::Real)
        for i in eachindex(xcurrent, xproposed)
            xproposed[i] = (rand() < p ? -1 : 1) * xcurrent[i]
        end
        return xproposed
    end
    function makeJ(n, p)
        n * p >= 2 || error("p must be large enough to ensure a connected graph")
        pbar = p * (1 - 1/n)
        function randconnect(i, j)
            j <= i && return false   # only fill in upper triangle
            j == i+1 && return true  # ensure graph is connected so we can use the Edwards bound
            return rand() < pbar
        end
        J = [randconnect(i, j) for i in 1:n, j in 1:n]
        J = J .| J'   # make symmetric
        return J
    end
    n, p = 10, 0.2
    J = makeJ(n, p)
    edwards_bound = -sum(J) / 4 - (n - 1) / 4
    method = SimulatedAnnealing(; neighbor=(xc, xp) -> maxcut_spinflip!(xc, xp, 2/n))
    options = Optim.Options(; iterations = 100_000)
    x0 = rand([-1.0, 1.0], n)
    # Ensure the initialization is worse than the Edwards bound
    while maxcut_objective(x0, J) <= edwards_bound
        x0 = rand([-1.0, 1.0], n)
    end
    results = Optim.optimize(x -> maxcut_objective(x, J), x0, method, options)
    xf = Optim.minimizer(results)
    @test maxcut_objective(xf, J) < edwards_bound
end
