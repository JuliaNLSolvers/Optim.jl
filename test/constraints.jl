using Optim, PositiveFactorizations
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

if VERSION >= v"0.5.0-dev+2396"
    macro inferred5(ex)
        Expr(:macrocall, Symbol("@inferred"), esc(ex))
    end
else
    macro inferred5(ex)
        esc(ex)
    end
end

@testset "Constraints" begin
    # Utility function for hand-setting the μ parameter
    function setstate!(state, μ, d, constraints, method)
        state.μ = μ
        Optim.update_fg!(d, constraints, state, method)
        Optim.update_h!(d, constraints, state, method)
    end

    @testset "Bounds parsing" begin
        b = @inferred5(Optim.ConstraintBounds([0.0, 0.5, 2.0], [1.0, 1.0, 2.0], [5.0, 3.8], [5.0, 4.0]))
        @test b.eqx == [3]
        @test b.valx == [2.0]
        @test b.ineqx == [1,2,2]
        @test b.σx == [-1,1,-1]
        @test b.bx == [1.0,0.5,1.0]
        @test b.iz == [1]
        @test b.σz == [1]
        @test b.eqc == [1]
        @test b.valc == [5]
        @test b.ineqc == [2,2]
        @test b.σc == [1,-1]
        @test b.bc == [3.8,4.0]
        io = IOBuffer()
        show(io, b)
        @test takebuf_string(io) == """
ConstraintBounds:
  Variables:
    x[3]=2.0
    x[1]≤1.0,x[2]≥0.5,x[2]≤1.0
    x[1]≥0.0
  Linear/nonlinear constraints:
    c_1=5.0
    c_2≥3.8,c_2≤4.0"""

        b = @inferred5(Optim.ConstraintBounds(Float64[], Float64[], [5.0, 3.8], [5.0, 4.0]))
        for fn in (:eqx, :valx, :ineqx, :σx, :bx, :iz, :σz)
            @test isempty(getfield(b, fn))
        end
        @test b.eqc == [1]
        @test b.valc == [5]
        @test b.ineqc == [2,2]
        @test b.σc == [1,-1]
        @test b.bc == [3.8,4.0]

        ba = Optim.ConstraintBounds([], [], [5.0, 3.8], [5.0, 4.0])
        @test eltype(ba) == Float64

        @test_throws ArgumentError Optim.ConstraintBounds([0.0, 0.5, 2.0], [1.0, 1.0, 2.0], [5.0, 4.8], [5.0, 4.0])
        @test_throws DimensionMismatch Optim.ConstraintBounds([0.0, 0.5, 2.0], [1.0, 1.0], [5.0, 4.8], [5.0, 4.0])
    end

    @testset "IPNewton computations" begin
        # Compare hand-computed gradient against that from automatic differentiation
        function check_autodiff(d, bounds, x, cfun::Function, bstate, μ)
            c = cfun(x)
            J = ForwardDiff.jacobian(cfun, x)
            p = Optim.pack_vec(x, bstate)
            ftot! = (p,storage)->Optim.lagrangian_fgvec!(p, storage, gx, bgrad, d, bounds, x, c, J, bstate, μ)
            pgrad = similar(p)
            ftot!(p, pgrad)
            chunksize = min(8, length(p))
            TD = ForwardDiff.Dual{chunksize,eltype(p)}
            xd = Array{TD}(length(x))
            bstated = Optim.BarrierStateVars{TD}(bounds)
            pcmp = similar(p)
            ftot = p->Optim.lagrangian_vec(p, d, bounds, xd, cfun, bstated, μ)
            ForwardDiff.gradient!(pcmp, ftot, p, ForwardDiff.Chunk{chunksize}())
            @test pcmp ≈ pgrad
        end
        # Basic setup
        μ = 0.2345678
        A = randn(3,3); H = A'*A
        d = TwiceDifferentiableFunction(x->(x'*H*x)[1]/2, (x,g)->(g[:] = H*x), (x,h)->(h[:,:]=H))
        x = broadcast(clamp, randn(3), -0.99, 0.99)
        gx = similar(x)
        cfun = x->Float64[]
        c = Float64[]
        J = Array{Float64}(0,0)
        options = OptimizationOptions()
        method = Optim.IPNewton()
        ## In the code, variable constraints are special-cased (for
        ## reasons of user-convenience and efficiency).  It's
        ## important to check that the special-casing yields the same
        ## result as the general case. So in the first three
        ## constrained cases below, we compare variable constraints
        ## against the same kind of constraint applied generically.
        cvar! = (x, c) -> copy!(c, x)
        cvarJ! = (x, J) -> copy!(J, eye(size(J)...))
        cvarh! = (x, λ, h) -> h  # h! adds to h, it doesn't replace it
        ## No constraints
        bounds = Optim.ConstraintBounds(Float64[], Float64[], Float64[], Float64[])
        bstate = Optim.BarrierStateVars(bounds, x)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, Float64[], Array{Float64}(0,0), bstate, μ)
        @test f_x == L == d.f(x)
        @test gx == H*x
        constraints = TwiceDifferentiableConstraintsFunction(
            (x,c)->nothing, (x,J)->nothing, (x,λ,H)->nothing, bounds)
        state = Optim.initial_state(method, options, d, constraints, x)
        @test state.gf ≈ gx
        @test state.Hf ≈ H
        ## Pure equality constraints on variables
        d = TwiceDifferentiableFunction(x->0.0, (x,g)->fill!(g, 0), (x,h)->fill!(h,0))
        xbar = fill(0.2, length(x))
        bounds = Optim.ConstraintBounds(xbar, xbar, [], [])
        bstate = Optim.BarrierStateVars(bounds)
        rand!(bstate.λxE)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, c, J, bstate, μ)
        @test f_x == 0
        @test L ≈ dot(bstate.λxE, xbar-x)
        @test gx == -bstate.λxE
        @test bgrad.λxE == xbar-x
        check_autodiff(d, bounds, x, cfun, bstate, μ)
        constraints = TwiceDifferentiableConstraintsFunction(
            (x,c)->nothing, (x,J)->nothing, (x,λ,H)->nothing, bounds)
        state = Optim.initial_state(method, options, d, constraints, x)
        copy!(state.bstate.λxE, bstate.λxE)
        setstate!(state, μ, d, constraints, method)
        @test state.gf ≈ [gx; xbar-x]
        n = length(x)
        @test state.Hf ≈ [eye(n,n) -eye(n,n); -eye(n,n) zeros(n,n)]
        # Now again using the generic machinery
        bounds = Optim.ConstraintBounds([], [], xbar, xbar)
        constraints = TwiceDifferentiableConstraintsFunction(cvar!, cvarJ!, cvarh!, bounds)
        state = Optim.initial_state(method, options, d, constraints, x)
        copy!(state.bstate.λcE, bstate.λxE)
        setstate!(state, μ, d, constraints, method)
        @test state.gf ≈ [gx; xbar-x]
        n = length(x)
        @test state.Hf ≈ [eye(n,n) -eye(n,n); -eye(n,n) zeros(n,n)]
        ## Nonnegativity constraints
        bounds = Optim.ConstraintBounds(zeros(length(x)), fill(Inf,length(x)), [], [])
        y = rand(length(x))
        bstate = Optim.BarrierStateVars(bounds, y)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, y, Float64[], Array{Float64}(0,0), bstate, μ)
        @test f_x == 0
        @test L ≈ -μ*sum(log, y)
        @test gx == -μ./y
        check_autodiff(d, bounds, y, cfun, bstate, μ)
        constraints = TwiceDifferentiableConstraintsFunction(
            (x,c)->nothing, (x,J)->nothing, (x,λ,H)->nothing, bounds)
        state = Optim.initial_state(method, options, d, constraints, y)
        setstate!(state, μ, d, constraints, method)
        @test state.gf ≈ -μ./y
        @test state.Hf ≈ μ*Diagonal(1./y.^2)
        # Now again using the generic machinery
        bounds = Optim.ConstraintBounds([], [], zeros(length(x)), fill(Inf,length(x)))
        constraints = TwiceDifferentiableConstraintsFunction(cvar!, cvarJ!, cvarh!, bounds)
        state = Optim.initial_state(method, options, d, constraints, y)
        setstate!(state, μ, d, constraints, method)
        @test state.gf ≈ -μ./y
        @test state.Hf ≈ μ*Diagonal(1./y.^2)
        ## General inequality constraints on variables
        lb, ub = rand(length(x))-2, rand(length(x))+1
        bounds = Optim.ConstraintBounds(lb, ub, [], [])
        bstate = Optim.BarrierStateVars(bounds, x)
        rand!(bstate.slack_x)  # intentionally displace from the correct value
        rand!(bstate.λx)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, Float64[], Array{Float64}(0,0), bstate, μ)
        @test f_x == 0
        s = bounds.σx .* (x[bounds.ineqx] - bounds.bx)
        Ltarget = -μ*sum(log, bstate.slack_x) +
            dot(bstate.λx, bstate.slack_x - s)
        @test L ≈ Ltarget
        dx = similar(gx); fill!(dx, 0)
        for (i,j) in enumerate(bounds.ineqx)
            dx[j] -= bounds.σx[i]*bstate.λx[i]
        end
        @test gx ≈ dx
        @test bgrad.slack_x == -μ./bstate.slack_x + bstate.λx
        check_autodiff(d, bounds, x, cfun, bstate, μ)
        constraints = TwiceDifferentiableConstraintsFunction(
            (x,c)->nothing, (x,J)->nothing, (x,λ,H)->nothing, bounds)
        state = Optim.initial_state(method, options, d, constraints, x)
        copy!(state.bstate.slack_x, bstate.slack_x)
        copy!(state.bstate.λx, bstate.λx)
        setstate!(state, μ, d, constraints, method)
        gxs, hxs = zeros(length(x)), zeros(length(x))
        s = state.bstate.slack_x
        for (i,j) in enumerate(bounds.ineqx)
            gxs[j] += -2*μ*bounds.σx[i]/s[i] + μ*(x[j]-bounds.bx[i])/s[i]^2
            hxs[j] += μ/s[i]^2
        end
        @test state.gf ≈ gxs
        @test state.Hf ≈ Diagonal(hxs)
        # Now again using the generic machinery
        bounds = Optim.ConstraintBounds([], [], lb, ub)
        constraints = TwiceDifferentiableConstraintsFunction(cvar!, cvarJ!, cvarh!, bounds)
        state = Optim.initial_state(method, options, d, constraints, x)
        copy!(state.bstate.slack_c, bstate.slack_x)
        copy!(state.bstate.λc, bstate.λx)
        setstate!(state, μ, d, constraints, method)
        @test state.gf ≈ gxs
        @test state.Hf ≈ Diagonal(hxs)
        ## Nonlinear equality constraints
        cfun = x->[x[1]^2+x[2]^2, x[2]*x[3]^2]
        cfun! = (x, c) -> copy!(c, cfun(x))
        cJ! = (x, J) -> copy!(J, [2*x[1] 2*x[2] 0;
                                  0 x[3]^2 2*x[2]*x[3]])
        ch! = function(x, λ, h)
            h[1,1] += 2*λ[1]
            h[2,2] += 2*λ[1]
            h[3,3] += 2*λ[2]*x[2]
        end
        c = cfun(x)
        J = ForwardDiff.jacobian(cfun, x)
        Jtmp = similar(J); @test cJ!(x, Jtmp) ≈ J  # just to check we did it right
        cbar = rand(length(c))
        bounds = Optim.ConstraintBounds([], [], cbar, cbar)
        bstate = Optim.BarrierStateVars(bounds, x, c)
        rand!(bstate.λcE)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, c, J, bstate, μ)
        @test f_x == 0
        @test L ≈ dot(bstate.λcE, cbar-c)
        @test gx ≈ -J'*bstate.λcE
        @test bgrad.λcE == cbar-c
        check_autodiff(d, bounds, x, cfun, bstate, μ)
        constraints = TwiceDifferentiableConstraintsFunction(cfun!, cJ!, ch!, bounds)
        state = Optim.initial_state(method, options, d, constraints, x)
        copy!(state.bstate.λcE, bstate.λcE)
        setstate!(state, μ, d, constraints, method)
        heq = zeros(length(x), length(x))
        ch!(x, bstate.λcE, heq)
        @test state.gf ≈ [gx; cbar-c]
        @test state.Hf ≈ [eye(length(x))-heq -J';
                          -J zeros(size(J,1), size(J,1))]
        ## Nonlinear inequality constraints
        bounds = Optim.ConstraintBounds([], [], rand(length(c))-1, rand(length(c))+1)
        bstate = Optim.BarrierStateVars(bounds, x, c)
        rand!(bstate.slack_c)  # intentionally displace from the correct value
        rand!(bstate.λc)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, c, J, bstate, μ)
        @test f_x == 0
        Ltarget = -μ*sum(log, bstate.slack_c) +
            dot(bstate.λc, bstate.slack_c - bounds.σc.*(c[bounds.ineqc]-bounds.bc))
        @test L ≈ Ltarget
        @test gx ≈ -J[bounds.ineqc,:]'*(bstate.λc.*bounds.σc)
        @test bgrad.slack_c == -μ./bstate.slack_c + bstate.λc
        @test bgrad.λc == bstate.slack_c - bounds.σc .* (c[bounds.ineqc] - bounds.bc)
        check_autodiff(d, bounds, x, cfun, bstate, μ)
        constraints = TwiceDifferentiableConstraintsFunction(cfun!, cJ!, ch!, bounds)
        state = Optim.initial_state(method, options, d, constraints, x)
        copy!(state.bstate.slack_c, bstate.slack_c)
        copy!(state.bstate.λc, bstate.λc)
        setstate!(state, μ, d, constraints, method)
        hineq = zeros(length(x), length(x))
        λ = zeros(size(J, 1))
        for (i,j) in enumerate(bounds.ineqc)
            λ[j] += bstate.λc[i]*bounds.σc[i]
        end
        ch!(x, λ, hineq)
        JI = J[bounds.ineqc,:]
        hxx = μ*JI'*Diagonal(1./bstate.slack_c.^2)*JI - hineq
        hp = full(cholfact(Positive, hxx))
        @test state.gf ≈ -JI'*(bounds.σc .* bstate.λc) + JI'*Diagonal(bounds.σc)*(bgrad.slack_c - μ*(bgrad.λc ./ bstate.slack_c.^2))
        @test state.Hf ≈ hp
    end

    @testset "IPNewton step" begin
        F = 1000
        d = TwiceDifferentiableFunction(x->F*x[1], (x,g) -> (g[1] = F), (x,h) -> (h[1,1] = 0))
        method = Optim.IPNewton()
        options = OptimizationOptions()
        μ = 1e-20
        x0 = μ/F*10  # minimum is at μ/F
        # Nonnegativity (the case that doesn't require slack variables)
        constraints = TwiceDifferentiableConstraintsFunction([0.0], [])
        state = Optim.initial_state(method, options, d, constraints, [x0])
        setstate!(state, μ, d, constraints, method)
        Optim.solve_step!(state, constraints)
        @test state.s[1] ≈ x0 - F*x0^2/μ
        qp = Optim.quadratic_parameters(constraints.bounds, state)
        @test qp[1] ≈ F*x0-μ*log(x0)
        @test qp[2] ≈ -(F-μ/x0)^2*x0^2/μ
        @test qp[3] ≈ μ/x0^2*(x0 - F*x0^2/μ)^2
        bstate, bstep, bounds = state.bstate, state.bstep, constraints.bounds
        αmax = Optim.estimate_maxstep(Inf, state.x[bounds.iz].*bounds.σz,
                                      state.s[bounds.iz].*bounds.σz)
        ϕ = α->Optim.lagrangian_linefunc(α, d, constraints, state)
        @test ϕ(0) ≈ qp[1]
        α, nf, ng = method.linesearch!(ϕ, 1.0, αmax, qp)
        @test α > 1e-3
    end

    @testset "Slack" begin
        σswap(σ, a, b) = σ == 1 ? (a, b) : (b, a)
        # Test that we achieve a high-precision minimum for fixed
        # μ. For anything other than nonnegativity/nonpositivity
        # constraints, this tests whether the slack variables are
        # solving the problem they were designed to address (the
        # possibility that adjacent floating-point numbers are too
        # widely spaced to accurately satisfy the KKT equations near a
        # boundary).
        F0 = 1000
        method = Optim.IPNewton()
        options = OptimizationOptions()
        μ = 1e-20   # smaller than eps(1.0)
        for σ in (1, -1)
            F = σ*F0
            # Nonnegativity/nonpositivity (the case that doesn't require slack variables)
            d = TwiceDifferentiableFunction(x->F*x[1], (x,g) -> (g[1] = F), (x,h) -> (h[1,1] = 0))
            constraints = TwiceDifferentiableConstraintsFunction(σswap(σ, [0.0], [])...)
            state = Optim.initial_state(method, options, d, constraints, [μ/F*10])
            setstate!(state, μ, d, constraints, method)
            for i = 1:10
                Optim.update_state!(d, constraints, state, method)
                Optim.update_fg!(d, constraints, state, method)
                Optim.update_h!(d, constraints, state, method)
            end
            @test state.x[1] ≈ μ/F
            # |x| ≥ 1, and check that we get slack precision better than eps(1.0)
            d = TwiceDifferentiableFunction(x->F*(x[1]-σ), (x,g) -> (g[1] = F), (x,h) -> (h[1,1] = 0))
            constraints = TwiceDifferentiableConstraintsFunction(σswap(σ, [Float64(σ)], [])...)
            state = Optim.initial_state(method, options, d, constraints, [(1+eps(1.0))*σ])
            setstate!(state, μ, d, constraints, method)
            for i = 1:10
                Optim.update_state!(d, constraints, state, method)
                Optim.update_fg!(d, constraints, state, method)
                Optim.update_h!(d, constraints, state, method)
            end
            @test state.x[1] == σ
            @test state.bstate.slack_x[1] ≈ μ/abs(F)
            # x >= 1 using the linear/nonlinear constraints
            d = TwiceDifferentiableFunction(x->F*(x[1]-σ), (x,g) -> (g[1] = F), (x,h) -> (h[1,1] = 0))
            constraints = TwiceDifferentiableConstraintsFunction(
                (x,c)->(c[1] = x[1]),
                (x,J)->(J[1,1] = 1.0),
                (x,λ,h)->nothing,
                [], [], σswap(σ, [Float64(σ)], [])...)
            state = Optim.initial_state(method, options, d, constraints, [(1+eps(1.0))*σ])
            setstate!(state, μ, d, constraints, method)
            for i = 1:10
                Optim.update_state!(d, constraints, state, method)
                Optim.update_fg!(d, constraints, state, method)
                Optim.update_h!(d, constraints, state, method)
            end
            @test state.x[1] == σ
            @test state.bstate.slack_c[1] ≈ μ/abs(F)
        end
    end
end

nothing
