@testset "interface" begin
    problem = Optim.UnconstrainedProblems.examples["Exponential"]
    f = problem.f
    g! = problem.g!
    h! = problem.h!
    nd = NonDifferentiable(f, zeros(problem.initial_x))
    od = OnceDifferentiable(f, g!, zeros(problem.initial_x))
    td = TwiceDifferentiable(f, g!, h!, zeros(problem.initial_x))
    tdref = TwiceDifferentiable(f, g!, h!, zeros(problem.initial_x))
    ref = optimize(tdref, problem.initial_x, Newton(), Optim.Options())
    # test AbstractObjective interface
    for obj in (nd, od, td)
        res = []
        push!(res, optimize(obj))
        # test that initial x isn't overwritten ref #491
        @test !(Optim.initial_state(res[end]) == Optim.minimizer(res[end]))
        push!(res, optimize(obj, problem.initial_x))

        push!(res, optimize(obj, problem.initial_x, Optim.Options()))

        push!(res, optimize(obj, Optim.Options()))
        # Test passing the objective, method and option, but no inital_x
        push!(res, optimize(obj, NelderMead(), Optim.Options()))
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

@testset "uninitialized interface" begin
    problem = Optim.UnconstrainedProblems.examples["Exponential"]
    f = problem.f
    g! = problem.g!
    h! = problem.h!
    nd = NonDifferentiable(f)
    od = OnceDifferentiable(f, g!)
    td = TwiceDifferentiable(f, g!, h!)
    ref = optimize(td, problem.initial_x, Newton(), Optim.Options())
    # test UninitializedObjective interface
    for obj in (nd, od, td)
        res = []
        push!(res, optimize(obj, problem.initial_x))

        push!(res, optimize(obj, problem.initial_x, Optim.Options()))

        push!(res, optimize(obj, problem.initial_x, Optim.Options()))
        # Test passing the objective, method and option, but no inital_x
        push!(res, optimize(obj, problem.initial_x, NelderMead(), Optim.Options()))
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


@testset "uninitialized objectives" begin
    # Test example
    function exponential(x::Vector)
        return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
    end

    function exponential_gradient!(storage::Vector, x::Vector)
        storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
        storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
    end
    function exponential_fg!(storage, x)
        exponential_gradient!(storage, x)
        exponential(x)
    end
    function exponential_hessian!(storage::Matrix, x::Vector)
        storage[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
        storage[1, 2] = 0.0
        storage[2, 1] = 0.0
        storage[2, 2] = 2.0 * exp((3.0 - x[1])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
    end

    x_seed = [0.0, 0.0]
    f_x_seed = 8157.682077608529

    und = NonDifferentiable(exponential)
    uod1 = OnceDifferentiable(exponential, exponential_gradient!)
    uod2 = OnceDifferentiable(exponential, exponential_gradient!, exponential_fg!)
    utd1 = TwiceDifferentiable(exponential, exponential_gradient!)
    utd2 = TwiceDifferentiable(exponential, exponential_gradient!, exponential_hessian!)
    utd3 = TwiceDifferentiable(exponential, exponential_gradient!, exponential_fg!, exponential_hessian!)
    nd = NonDifferentiable(und, x_seed)
    od1 = OnceDifferentiable(uod1, x_seed)
    od2 = OnceDifferentiable(uod2, x_seed)
    td1 =  TwiceDifferentiable(utd1, x_seed)
    td2 = TwiceDifferentiable(utd2, x_seed)
    td3 = TwiceDifferentiable(utd3, x_seed)
end
