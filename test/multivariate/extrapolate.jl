import LineSearches

@testset "Extrapolation" begin
    methods = [LBFGS(),
               ConjugateGradient(),
               LBFGS(alphaguess = LineSearches.InitialQuadratic(),
                     linesearch = LineSearches.BackTracking(order=2))]
    msgs = ["LBFGS Default Options: ",
            "CG Default Options: ",
            "LBFGS + Backtracking + Extrapolation: "]

    if debug_printing
        println("--------------------")
        println("Rosenbrock Example: ")
        println("--------------------")
    end
    rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    for (method, msg) in zip(methods, msgs)
        results = Optim.optimize(rosenbrock, zeros(2), method)
        debug_printing && println(msg, "g_calls = ", results.g_calls, ", f_calls = ", results.f_calls)
    end

    if debug_printing
        println("--------------------------------------")
        println("p-Laplacian Example (preconditioned): ")
        println("--------------------------------------")
    end
    plap(U; n=length(U)) = (n-1) * sum((0.1 .+ diff(U).^2).^2) - sum(U) / (n-1)
    plap1(U; n=length(U), dU = diff(U), dW = 4 .* (0.1 .+ dU.^2) .* dU) =
                            (n - 1) .* ([0.0; dW] .- [dW; 0.0]) .- ones(n) / (n - 1)
    precond(x::Vector) = precond(length(x))
    precond(n::Number) = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n),  1 => -ones(n-1)) * (n+1)
    f(X) = plap([0;X;0])
    g!(g, X) = copyto!(g, (plap1([0;X;0]))[2:end-1])
    N = 100
    initial_x = zeros(N)
    P = precond(initial_x)
    methods = [LBFGS(P=P),
               ConjugateGradient(P=P),
               LBFGS(alphaguess = LineSearches.InitialQuadratic(),
                     linesearch = LineSearches.BackTracking(order=2), P=P)]

    for (method, msg) in zip(methods, msgs)
        results = Optim.optimize(f, g!, copy(initial_x), method)
        debug_printing && println(msg, "g_calls = ", Optim.g_calls(results), ", f_calls = ", Optim.f_calls(results))
    end
end
