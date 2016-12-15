let
    # Quadratic objective function
    # For (A*x-b)^2/2
    function quadratic!(x, g, AtA, Atb, tmp)
        calc_grad = !(g === nothing)
        A_mul_B!(tmp, AtA, x)
        v = dot(x,tmp)/2 + dot(Atb,x)
        if calc_grad
            for i = 1:length(g)
                g[i] = tmp[i] + Atb[i]
            end
        end
        return v
    end

    srand(1)
    N = 8
    boxl = 2.0
    outbox = false
    # Generate a problem where the bounds-free solution lies outside of the chosen box
    global objective
    while !outbox
        A = randn(N,N)
        AtA = A'*A
        b = randn(N)
        initial_x = randn(N)
        tmp = similar(initial_x)
        func = (x, g) -> quadratic!(x, g, AtA, A'*b, tmp)
        objective = Optim.DifferentiableFunction(x->func(x, nothing), (x,g)->func(x,g), func)
        results = Optim.optimize(objective, initial_x, ConjugateGradient())
        results = Optim.optimize(objective, Optim.minimizer(results), ConjugateGradient())  # restart to ensure high-precision convergence
        @test Optim.converged(results)
        g = similar(initial_x)
        @test func(Optim.minimizer(results), g) + dot(b,b)/2 < 1e-8
        @test norm(g) < 1e-4
        outbox = any(t -> abs(t) .> boxl, Optim.minimizer(results))
    end

    # fminbox
    l = fill(-boxl, N)
    u = fill(boxl, N)
    initial_x = (rand(N)-0.5)*boxl
    for _optimizer in (ConjugateGradient, GradientDescent, LBFGS, BFGS)
        results = Optim.optimize(objective, initial_x, l, u, Fminbox(), optimizer = _optimizer)
        @test Optim.converged(results)

        g = similar(initial_x)
        objective.fg!(Optim.minimizer(results), g)
        for i = 1:N
            @test abs(g[i]) < 3e-3 || (Optim.minimizer(results)[i] < -boxl+1e-3 && g[i] > 0) || (Optim.minimizer(results)[i] > boxl-1e-3 && g[i] < 0)
        end
    end

    # tests for #180
    results = Optim.optimize(objective, initial_x, l, u, Fminbox(); iterations = 2)
    @test Optim.iterations(results) == 2
    @test Optim.minimum(results) == objective.f(Optim.minimizer(results))

    # might fail if changes are made to Optim.jl
    # TODO: come up with a better test
    results = Optim.optimize(objective, initial_x, l, u, Fminbox(); optimizer_o = OptimizationOptions(iterations = 2))
    @test Optim.iterations(results) == 470
end
