# The unscaled method, the default up to Optim 2
const UNSCALED = (initial_stepnorm = nothing, initial_scaling = nothing, self_scaling = nothing)

@testset "BFGS" begin
    # Trigonometric ends up in a local minimum
    skip = ("Trigonometric",)
    run_optim_tests(
        BFGS();
        skip,
        show_name = debug_printing,
    )
    run_optim_tests(
        BFGS(; UNSCALED...);
        skip,
        show_name = debug_printing,
    )
end

@testset "BFGS defaults" begin
    m = BFGS()
    @test m.initial_stepnorm == 1
    @test m.initial_scaling isa Optim.ShannoPhua
    @test m.self_scaling isa Optim.AlBaali
    # the first trial step has sup-norm 1 by default, and is -g without scaling
    f(x) = sum(abs2, x .- [3.0, -4.0]) / 2
    g!(G, x) = (G .= x .- [3.0, -4.0])
    opts = Optim.Options(iterations = 1, store_trace = true, extended_trace = true)
    H1 = Optim.trace(optimize(f, g!, zeros(2), BFGS(), opts))[1].metadata["~inv(H)"]
    @test H1 ≈ I(2) / 4
    H1 = Optim.trace(optimize(f, g!, zeros(2), BFGS(; UNSCALED...), opts))[1].metadata["~inv(H)"]
    @test H1 ≈ I(2)
end

@testset "BFGS with Hessian scaling" begin
    skip = ("Trigonometric",)
    # ShannoPhua alone needs 1906 and 1149 iterations on these two ill-conditioned quadratics
    run_optim_tests(
        BFGS(; UNSCALED..., initial_scaling = Optim.ShannoPhua());
        skip,
        iteration_exceptions = (("Paraboloid Diagonal", 3000), ("Paraboloid Random Matrix", 2000)),
        show_name = debug_printing,
    )
    for method in (BFGS(; UNSCALED..., self_scaling = Optim.AlBaali()),
                   BFGS(; UNSCALED..., self_scaling = Optim.OrenLuenberger()),
                   BFGS(initial_stepnorm = nothing))
        run_optim_tests(method; skip, show_name = debug_printing)
    end
end

# The inverse BFGS update of H with step s and gradient change y
bfgs_update(H, s, y) = (ρ = 1 / dot(s, y); V = I - ρ * y * s'; V' * H * V + ρ * s * s')

@testset "BFGS scaling factors" begin
    a = [1.0, 10.0, 100.0]
    f(x) = sum(a .* x .^ 2) / 2
    g!(G, x) = (G .= a .* x)
    opts = Optim.Options(iterations = 3, g_abstol = 0.0, store_trace = true, extended_trace = true)
    steps(tr, k) = (s = tr[k+1].metadata["x"] - tr[k].metadata["x"]; (s, a .* s))

    # ShannoPhua: the first update starts from (s'y / y'y) I instead of I
    tr = Optim.trace(optimize(f, g!, ones(3), BFGS(initial_scaling = Optim.ShannoPhua(), self_scaling = nothing), opts))
    s, y = steps(tr, 1)
    @test tr[2].metadata["~inv(H)"] ≈ bfgs_update(dot(s, y) / dot(y, y) * I(3), s, y)
    s, y = steps(tr, 2)
    @test tr[3].metadata["~inv(H)"] ≈ bfgs_update(tr[2].metadata["~inv(H)"], s, y)

    # self_scaling: H is divided by τ(ρ), ρ = y's / s'Bs, before every update.
    # From H0 = c I the first step is along -g = -a, so ρ = c a'A a / a'a ≈ 99.1 c ≈ 0.69,
    # which AlBaali scales and AlBaali(cutoff = 0.8) leaves alone.
    H0 = x -> Matrix(0.007 * I(3))
    for (rule, τfun) in ((Optim.AlBaali(), ρ -> 1 / 2 <= ρ < 1 ? ρ : 1.0),
                         (Optim.AlBaali(cutoff = 0.8), ρ -> 0.8 <= ρ < 1 ? ρ : 1.0),
                         (Optim.OrenLuenberger(), ρ -> ρ))
        tr = Optim.trace(optimize(f, g!, ones(3), BFGS(self_scaling = rule, initial_invH = H0), opts))
        for k in 1:3
            H = tr[k].metadata["~inv(H)"]
            s, y = steps(tr, k)
            τ = τfun(dot(s, y) / dot(s, H \ s))
            @test tr[k+1].metadata["~inv(H)"] ≈ bfgs_update(H / τ, s, y)
        end
    end
    ρ1 = let tr = Optim.trace(optimize(f, g!, ones(3), BFGS(initial_invH = H0), opts))
        s, y = steps(tr, 1)
        dot(s, y) / dot(s, tr[1].metadata["~inv(H)"] \ s)
    end
    @test 0.5 <= ρ1 < 0.8

    # with both, the first update uses the initial scaling and the later ones self-scaling
    tr = Optim.trace(optimize(f, g!, ones(3),
                              BFGS(initial_scaling = Optim.ShannoPhua(), self_scaling = Optim.AlBaali()), opts))
    s, y = steps(tr, 1)
    @test tr[2].metadata["~inv(H)"] ≈ bfgs_update(dot(s, y) / dot(y, y) * I(3), s, y)

    # ShannoPhua does not touch a user-supplied initial inverse Hessian
    Hd = x -> Matrix(Diagonal(1 ./ a))
    r1 = optimize(f, g!, ones(3), BFGS(initial_invH = Hd, initial_scaling = Optim.ShannoPhua()), opts)
    r2 = optimize(f, g!, ones(3), BFGS(initial_invH = Hd), opts)
    @test Optim.minimizer(r1) == Optim.minimizer(r2)

    # complex inputs: the scaled identity takes the element type of x
    A = Diagonal([1.0, 10.0, 100.0])
    b = ComplexF64[1 + 2im, -1im, 3]
    fc(z) = real(dot(z, A * z)) / 2 - real(dot(b, z))
    gc!(G, z) = (G .= A * z .- b)
    rc = optimize(fc, gc!, zeros(ComplexF64, 3),
                  BFGS(initial_scaling = Optim.ShannoPhua(), self_scaling = Optim.AlBaali()))
    @test Optim.converged(rc)
    @test Optim.minimizer(rc) ≈ A \ b
end
