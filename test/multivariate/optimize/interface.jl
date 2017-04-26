@testset "interface" begin
    problem = Optim.UnconstrainedProblems.examples["Exponential"]
    f = problem.f
    g! = problem.g!
    h! = problem.h!
    nd = NonDifferentiable(f, problem.initial_x)
    od = OnceDifferentiable(f, g!, problem.initial_x)
    td = TwiceDifferentiable(f, g!, h!, problem.initial_x)
    ref = optimize(td, problem.initial_x, Newton(), Optim.Options())
    # test AbstractObjective interface
    for obj in (nd, od, td)
        res = []
        push!(res, optimize(obj))
        push!(res, optimize(obj, problem.initial_x))

        push!(res, optimize(obj, problem.initial_x, Optim.Options()))

        push!(res, optimize(obj, Optim.Options()))
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
end
