@testset "interface" begin
    problem = MultivariateProblems.UnconstrainedProblems.examples["Exponential"]
    f = MVP.objective(problem)
    g! = MVP.gradient(problem)
    h! = MVP.hessian(problem)
    nd = NonDifferentiable(f, fill!(similar(problem.initial_x), 0))
    od = OnceDifferentiable(f, g!, fill!(similar(problem.initial_x), 0))
    td = TwiceDifferentiable(f, g!, h!, fill!(similar(problem.initial_x), 0))
    tdref = TwiceDifferentiable(f, g!, h!, fill!(similar(problem.initial_x), 0))
    ref = optimize(tdref, problem.initial_x, Newton(), Optim.Options())
    # test AbstractObjective interface
    for obj in (nd, od, td)
        res = []
        push!(res, optimize(obj, problem.initial_x))

        push!(res, optimize(obj, problem.initial_x, Optim.Options()))

        for r in res
            @test norm(Optim.minimum(ref)-Optim.minimum(r)) < 1e-6
        end
    end
    ad_res = optimize(od, problem.initial_x, Newton())
    @test norm(Optim.minimum(ref)-Optim.minimum(ad_res)) < 1e-6
    ad_res2 = optimize(od, problem.initial_x, Newton())
    @test norm(Optim.minimum(ref)-Optim.minimum(ad_res2)) < 1e-6
    # test f, g!, h! interface
    for tup in ((f,), (f, g!), (f, g!, h!))
        fgh_res = []
        push!(fgh_res, optimize(tup..., problem.initial_x))
        for m in (NelderMead(), LBFGS(), Newton())
            push!(fgh_res, optimize(tup..., problem.initial_x; f_tol = 1e-8))
            push!(fgh_res, optimize(tup..., problem.initial_x, m))
            push!(fgh_res, optimize(tup..., problem.initial_x, m, Optim.Options()))
        end
        for r in fgh_res
            @test norm(Optim.minimum(ref)-Optim.minimum(r)) < 1e-6
        end
    end
    # simple tests for https://github.com/JuliaNLSolvers/Optim.jl/issues/805
    @test AcceleratedGradientDescent(alphaguess=1.0).alphaguess! isa Optim.LineSearches.InitialStatic
    @test BFGS(alphaguess=1.0).alphaguess! isa Optim.LineSearches.InitialStatic
    @test ConjugateGradient(alphaguess=1.0).alphaguess! isa Optim.LineSearches.InitialStatic
    @test GradientDescent(alphaguess=1.0).alphaguess! isa Optim.LineSearches.InitialStatic
    @test MomentumGradientDescent(alphaguess=1.0).alphaguess! isa Optim.LineSearches.InitialStatic
    @test Newton(alphaguess=1.0).alphaguess! isa Optim.LineSearches.InitialStatic
    optimize(od, problem.initial_x, AcceleratedGradientDescent(alphaguess=1.0))
    optimize(od, problem.initial_x, BFGS(alphaguess=1.0))
    optimize(od, problem.initial_x, ConjugateGradient(alphaguess=1.0))
    optimize(od, problem.initial_x, GradientDescent(alphaguess=1.0))
    optimize(od, problem.initial_x, MomentumGradientDescent(alphaguess=1.0))
    optimize(td, problem.initial_x, Newton(alphaguess=1.0))
end

@testset "only_fg!, only_fgh!" begin
    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    function g!(G, x)
      G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
      G[2] = 200.0 * (x[2] - x[1]^2)
      G
    end
    function h!(H, x)
      H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
      H[1, 2] = -400.0 * x[1]
      H[2, 1] = -400.0 * x[1]
      H[2, 2] = 200.0
      H
    end
    function fg!(F,G,x)
      G === nothing || g!(G,x)
      F === nothing || return f(x)
      nothing
    end
    function fgh!(F,G,H,x)
      G === nothing || g!(G,x)
      H === nothing || h!(H,x)
      F === nothing || return f(x)
      nothing
    end

    result_fg! = Optim.optimize(Optim.only_fg!(fg!), [0., 0.], Optim.LBFGS()) # works fine
    @test result_fg!.minimizer ≈ [1,1]
    result_fgh! = Optim.optimize(Optim.only_fgh!(fgh!), [0., 0.], Optim.Newton())
    @test result_fgh!.minimizer ≈ [1,1]
end


@testset "#816" begin
    w = rand(2)
    f(x) = sum(x.^2)
    g!(G, x) = @. G = 2x
    g(x) = 2x
    h!(H, x) = @. H = [2.0 0.0; 0.0 2.0]
    hv!(Hv, x) = @. Hv = [2.0, 2.0] .* x
    _hv!(Hv, x, v) = @. Hv = [2.0, 2.0] .* x
    res = Optim.optimize(f, w)
    @test res.method isa NelderMead

    res = Optim.optimize(f, g!, w)
    @test res.method isa LBFGS
    function fg!(_, G, x)
        isnothing(G) || g!(G, x)
        return f(x)
    end
    function fg(x)
        return f(x), g(x)
    end

    res = Optim.optimize(Optim.only_fg!(fg!), w)
    @test res.method isa LBFGS

    res = Optim.optimize(Optim.only_fg(fg), w)
    @test res.method isa LBFGS

    res = Optim.optimize(Optim.only_g_and_fg(g, fg), w)
    @test res.method isa LBFGS

    function fgh!(_, G, H, x)
        isnothing(G) || g!(G, x)
        isnothing(H) || h!(H, x)
        return f(x)
    end

    res = Optim.optimize(Optim.only_fgh!(fgh!), w)
    @test res.method isa Newton

    res = Optim.optimize(Optim.only_fgh!(fgh!), w)
    @test res.method isa Newton

    function fghv!(_, G, Hv, x, v)
        isnothing(G) || g!(G, x)
        isnothing(Hv) || hv!(Hv, v)
        return f(x)
    end

    res = Optim.optimize(Optim.only_fghv!(fghv!), w)
    @test res.method isa Optim.KrylovTrustRegion

    res = Optim.optimize(Optim.only_fg_and_hv!(fg!, _hv!), w)
    @test res.method isa Optim.KrylovTrustRegion

end
