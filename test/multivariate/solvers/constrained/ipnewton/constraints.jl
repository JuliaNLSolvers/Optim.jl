@testset "Constraints" begin
    # Utility function for hand-setting the μ parameter
    function setstate!(state, μ, d, constraints, method)
        state.μ = μ
        Optim.update_fg!(d, constraints, state, method)
        Optim.update_h!(d, constraints, state, method)
    end

    @testset "Bounds parsing" begin
        b = Optim.ConstraintBounds([0.0, 0.5, 2.0], [1.0, 1.0, 2.0], [5.0, 3.8], [5.0, 4.0])
        @test b.eqx == [3]
        @test b.valx == [2.0]
        @test b.ineqx == [1,1,2,2]
        @test b.σx == [1,-1,1,-1]
        @test b.bx == [0.0,1.0,0.5,1.0]
        @test b.eqc == [1]
        @test b.valc == [5]
        @test b.ineqc == [2,2]
        @test b.σc == [1,-1]
        @test b.bc == [3.8,4.0]
        io = IOBuffer()
        show(io, b)
        @test String(take!(io)) == "ConstraintBounds:\n  Variables:\n    x[3]=2.0\n    x[1]≥0.0, x[1]≤1.0, x[2]≥0.5, x[2]≤1.0\n  Linear/nonlinear constraints:\n    c_1=5.0\n    c_2≥3.8, c_2≤4.0"

        b = Optim.ConstraintBounds(Float64[], Float64[], [5.0, 3.8], [5.0, 4.0])
        for fn in (:eqx, :valx, :ineqx, :σx, :bx)
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
            ftot! = (storage, p)->Optim.lagrangian_fgvec!(p, storage, gx, bgrad, d, bounds, x, c, J, bstate, μ)
            pgrad = similar(p)
            ftot!(pgrad, p)
            chunksize = min(8, length(p))
            TD = ForwardDiff.Dual{ForwardDiff.Tag{Nothing,Float64},eltype(p),chunksize}
            xd = similar(x, TD)
            bstated = Optim.BarrierStateVars{TD}(bounds)
            pcmp = similar(p)
            ftot = p->Optim.lagrangian_vec(p, d, bounds, xd, cfun, bstated, μ)
            #ForwardDiff.gradient!(pcmp, ftot, p, ForwardDiff.{chunksize}())
            ForwardDiff.gradient!(pcmp, ftot, p)
            @test pcmp ≈ pgrad
        end
        # Basic setup (using two objectives, one equal to zero and the other a Gaussian)
        μ = 0.2345678
        d0 = TwiceDifferentiable(x->0.0, (g,x)->fill!(g, 0.0), (h,x)->fill!(h,0), rand(3))
        A = randn(3,3); H = A'*A
        dg = TwiceDifferentiable(x->(x'*H*x)[1]/2, (g,x)->(g[:] = H*x), (h,x)->(h[:,:]=H), rand(3))
        x = clamp.(randn(3), -0.99, 0.99)
        gx = similar(x)
        cfun = x->Float64[]
        c = Float64[]
        J = Array{Float64}(undef, 0,0)
        method = Optim.IPNewton(μ0 = μ)
        options = Optim.Options(; Optim.default_options(method)...)
        ## In the code, variable constraints are special-cased (for
        ## reasons of user-convenience and efficiency).  It's
        ## important to check that the special-casing yields the same
        ## result as the general case. So in the first three
        ## constrained cases below, we compare variable constraints
        ## against the same kind of constraint applied generically.
        cvar! = (c, x) -> copyto!(c, x)
        cvarJ! = (J, x) -> copyto!(J, Matrix{Float64}(I, size(J)...))
        cvarh! = (h, x, λ) -> h  # h! adds to h, it doesn't replace it

        ## No constraints
        bounds = Optim.ConstraintBounds(Float64[], Float64[], Float64[], Float64[])
        bstate = Optim.BarrierStateVars(bounds, x)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, dg, bounds, x, Float64[], Array{Float64}(undef, 0,0), bstate, μ)
        @test f_x == L == dg.f(x)
        @test gx == H*x
        constraints = TwiceDifferentiableConstraints(
            (c,x)->nothing, (J,x)->nothing, (H,x,λ)->nothing, bounds)
        state = Optim.initial_state(method, options, dg, constraints, x)
        @test Optim.gf(bounds, state) ≈ gx
        @test Optim.Hf(constraints, state) ≈ H
        stateconvert = convert(Optim.IPNewtonState{Float64, Vector{Float64}}, state)
        @test Optim.gf(bounds, stateconvert) ≈ gx
        @test Optim.Hf(constraints, stateconvert) ≈ H
        ## Pure equality constraints on variables
        xbar = fill(0.2, length(x))
        bounds = Optim.ConstraintBounds(xbar, xbar, [], [])
        bstate = Optim.BarrierStateVars(bounds)
        rand!(bstate.λxE)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d0, bounds, x, c, J, bstate, μ)
        @test f_x == 0
        @test L ≈ dot(bstate.λxE, xbar-x)
        @test gx == -bstate.λxE
        @test bgrad.λxE == xbar-x


        # TODO: Fix autodiff check
        #check_autodiff(d0, bounds, x, cfun, bstate, μ)



        constraints = TwiceDifferentiableConstraints(
            (c,x)->nothing, (J,x)->nothing, (H,x,λ)->nothing, bounds)
        state = Optim.initial_state(method, options, d0, constraints, x)
        copyto!(state.bstate.λxE, bstate.λxE)
        setstate!(state, μ, d0, constraints, method)
        @test Optim.gf(bounds, state) ≈ [gx; xbar-x]
        n = length(x)
        eyen = Matrix{Float64}(I, n, n)
        @test Optim.Hf(constraints, state) ≈ [eyen -eyen; -eyen zeros(n,n)]
        # Now again using the generic machinery
        bounds = Optim.ConstraintBounds([], [], xbar, xbar)
        constraints = TwiceDifferentiableConstraints(cvar!, cvarJ!, cvarh!, bounds)
        state = Optim.initial_state(method, options, d0, constraints, x)
        copyto!(state.bstate.λcE, bstate.λxE)
        setstate!(state, μ, d0, constraints, method)
        @test Optim.gf(bounds, state) ≈ [gx; xbar-x]
        n = length(x)
        eyen = Matrix{Float64}(I, n, n)
        @test Optim.Hf(constraints, state) ≈ [eyen -eyen; -eyen zeros(n,n)]
        ## Nonnegativity constraints
        bounds = Optim.ConstraintBounds(zeros(length(x)), fill(Inf,length(x)), [], [])
        y = rand(length(x))
        bstate = Optim.BarrierStateVars(bounds, y)
        rand!(bstate.λx)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d0, bounds, y, Float64[], Array{Float64}(undef, 0,0), bstate, μ)
        @test f_x == 0
        @test L ≈ -μ*sum(log, y)
        @test bgrad.slack_x == -μ./y + bstate.λx
        @test gx == -bstate.λx
        # TODO: Fix autodiff check
        #check_autodiff(d0, bounds, y, cfun, bstate, μ)
        constraints = TwiceDifferentiableConstraints(
            (c,x)->nothing, (J,x)->nothing, (H,x,λ)->nothing, bounds)
        state = Optim.initial_state(method, options, d0, constraints, y)
        setstate!(state, μ, d0, constraints, method)
        @test Optim.gf(bounds, state) ≈ -μ./y
        @test Optim.Hf(constraints, state) ≈ μ*Diagonal(1 ./ y.^2)
        # Now again using the generic machinery
        bounds = Optim.ConstraintBounds([], [], zeros(length(x)), fill(Inf,length(x)))
        constraints = TwiceDifferentiableConstraints(cvar!, cvarJ!, cvarh!, bounds)
        state = Optim.initial_state(method, options, d0, constraints, y)
        setstate!(state, μ, d0, constraints, method)
        @test Optim.gf(bounds, state) ≈ -μ./y
        @test Optim.Hf(constraints, state) ≈ μ*Diagonal(1 ./ y.^2)
        ## General inequality constraints on variables
        lb, ub = rand(length(x)).-2, rand(length(x)).+1
        bounds = Optim.ConstraintBounds(lb, ub, [], [])
        bstate = Optim.BarrierStateVars(bounds, x)
        rand!(bstate.slack_x)  # intentionally displace from the correct value
        rand!(bstate.λx)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d0, bounds, x, Float64[], Array{Float64}(undef, 0,0), bstate, μ)
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
        # TODO: Fix autodiff check
        #check_autodiff(d0, bounds, x, cfun, bstate, μ)
        constraints = TwiceDifferentiableConstraints(
            (c,x)->nothing, (J,x)->nothing, (H,x,λ)->nothing, bounds)
        state = Optim.initial_state(method, options, d0, constraints, x)
        copyto!(state.bstate.slack_x, bstate.slack_x)
        copyto!(state.bstate.λx, bstate.λx)
        setstate!(state, μ, d0, constraints, method)
        gxs, hxs = zeros(length(x)), zeros(length(x))
        s, λ = state.bstate.slack_x, state.bstate.λx
        for (i,j) in enumerate(bounds.ineqx)
            # # Primal
            # gxs[j] += -2*μ*bounds.σx[i]/s[i] + μ*(x[j]-bounds.bx[i])/s[i]^2
            # hxs[j] += μ/s[i]^2
            # Primal-dual
            gstmp = -μ/s[i] + λ[i]
            gλtmp = s[i] - bounds.σx[i]*(x[j]-bounds.bx[i])
            htmp = λ[i]/s[i]
            hxs[j] += htmp
            gxs[j] += bounds.σx[i]*(gstmp - λ[i]) - bounds.σx[i]*htmp*gλtmp
        end
        @test Optim.gf(bounds, state) ≈ gxs
        @test Optim.Hf(constraints, state) ≈ Diagonal(hxs)
        # Now again using the generic machinery
        bounds = Optim.ConstraintBounds([], [], lb, ub)
        constraints = TwiceDifferentiableConstraints(cvar!, cvarJ!, cvarh!, bounds)
        state = Optim.initial_state(method, options, d0, constraints, x)
        copyto!(state.bstate.slack_c, bstate.slack_x)
        copyto!(state.bstate.λc, bstate.λx)
        setstate!(state, μ, d0, constraints, method)
        @test Optim.gf(bounds, state) ≈ gxs
        @test Optim.Hf(constraints, state) ≈ Diagonal(hxs)
        ## Nonlinear equality constraints
        cfun = x->[x[1]^2+x[2]^2, x[2]*x[3]^2]
        cfun! = (c, x) -> copyto!(c, cfun(x))
        cJ! = (J, x) -> copyto!(J, [2*x[1] 2*x[2] 0;
                                  0 x[3]^2 2*x[2]*x[3]])
        ch! = function(h, x, λ)
            h[1,1] += 2*λ[1]
            h[2,2] += 2*λ[1]
            h[3,3] += 2*λ[2]*x[2]
        end
        c = cfun(x)
        J = ForwardDiff.jacobian(cfun, x)
        Jtmp = similar(J); @test cJ!(Jtmp, x) ≈ J  # just to check we did it right
        cbar = rand(length(c))
        bounds = Optim.ConstraintBounds([], [], cbar, cbar)
        bstate = Optim.BarrierStateVars(bounds, x, c)
        rand!(bstate.λcE)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d0, bounds, x, c, J, bstate, μ)
        @test f_x == 0
        @test L ≈ dot(bstate.λcE, cbar-c)
        @test gx ≈ -J'*bstate.λcE
        @test bgrad.λcE == cbar-c
        # TODO: Fix autodiff check
        #check_autodiff(d0, bounds, x, cfun, bstate, μ)
        constraints = TwiceDifferentiableConstraints(cfun!, cJ!, ch!, bounds)
        state = Optim.initial_state(method, options, d0, constraints, x)
        copyto!(state.bstate.λcE, bstate.λcE)
        setstate!(state, μ, d0, constraints, method)
        heq = zeros(length(x), length(x))
        ch!(heq, x, bstate.λcE)
        @test Optim.gf(bounds, state) ≈ [gx; cbar-c]
        @test Optim.Hf(constraints, state) ≈ [Matrix(cholesky(Positive, heq)) -J';
                                                  -J zeros(size(J,1), size(J,1))]
        ## Nonlinear inequality constraints
        bounds = Optim.ConstraintBounds([], [], .-rand(length(c)).-1, rand(length(c)).+2)
        bstate = Optim.BarrierStateVars(bounds, x, c)
        rand!(bstate.slack_c)  # intentionally displace from the correct value
        rand!(bstate.λc)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d0, bounds, x, c, J, bstate, μ)
        @test f_x == 0
        Ltarget = -μ*sum(log, bstate.slack_c) +
            dot(bstate.λc, bstate.slack_c - bounds.σc.*(c[bounds.ineqc]-bounds.bc))
        @test L ≈ Ltarget
        @test gx ≈ -J[bounds.ineqc,:]'*(bstate.λc.*bounds.σc)
        @test bgrad.slack_c == -μ./bstate.slack_c + bstate.λc
        @test bgrad.λc == bstate.slack_c - bounds.σc .* (c[bounds.ineqc] - bounds.bc)
        # TODO: Fix autodiff check
        #check_autodiff(d0, bounds, x, cfun, bstate, μ)
        constraints = TwiceDifferentiableConstraints(cfun!, cJ!, ch!, bounds)
        state = Optim.initial_state(method, options, d0, constraints, x)
        copyto!(state.bstate.slack_c, bstate.slack_c)
        copyto!(state.bstate.λc, bstate.λc)
        setstate!(state, μ, d0, constraints, method)
        hineq = zeros(length(x), length(x))
        λ = zeros(size(J, 1))
        for (i,j) in enumerate(bounds.ineqc)
            λ[j] += bstate.λc[i]*bounds.σc[i]
        end
        ch!(hineq, x, λ)
        JI = J[bounds.ineqc,:]
        # # Primal
        # hxx = μ*JI'*Diagonal(1 ./ bstate.slack_c.^2)*JI - hineq
        # gf = -JI'*(bounds.σc .* bstate.λc) + JI'*Diagonal(bounds.σc)*(bgrad.slack_c - μ(bgrad.λc ./ bstate.slack_c.^2))
        # Primal-dual
        #        hxx = full(cholesky(Positive, -hineq)) + JI'*Diagonal(bstate.λc./bstate.slack_c)*JI
        hxx = -hineq + JI'*Diagonal(bstate.λc./bstate.slack_c)*JI
        gf = -JI'*(bounds.σc .* bstate.λc) + JI'*Diagonal(bounds.σc)*(bgrad.slack_c - (bgrad.λc .* bstate.λc ./ bstate.slack_c))
        @test Optim.gf(bounds, state) ≈ gf
        @test Optim.Hf(constraints, state) ≈ Matrix(cholesky(Positive, hxx, Val{true}))
    end

    @testset "IPNewton initialization" begin
        method = IPNewton()
        options = Optim.Options(; Optim.default_options(method)...)
        x = [1.0,0.1,0.3,0.4]
        ## A linear objective function (hessian is zero)
        f_g = [1.0,2.0,3.0,4.0]
        d = TwiceDifferentiable(x->dot(x, f_g), (g,x)->copyto!(g, f_g), (h,x)->fill!(h, 0), x)
        # Variable bounds
        constraints = TwiceDifferentiableConstraints([0.5, 0.0, -Inf, -Inf], [Inf, Inf, 1.0, 0.8])
        state = Optim.initial_state(method, options, d, constraints, x)
        Optim.update_fg!(d, constraints, state, method)
        @test norm(f_g - state.g) ≈ 0.01*norm(f_g)
        # Nonlinear inequalities
        constraints = TwiceDifferentiableConstraints(
            (c,x)->(c[1]=x[1]*x[2]; c[2]=3*x[3]+x[4]^2),
            (J,x)->(J[:,:] = [x[2] x[1] 0 0; 0 0 3 2*x[4]]),
            (h,x,λ)->(h[4,4] += λ[2]*2),
            [], [], [0.05, 0.4], [0.15, 4.4])
        @test Optim.isinterior(constraints, x)
        state = Optim.initial_state(method, options, d, constraints, x)
        Optim.update_fg!(d, constraints, state, method)
        @test norm(f_g - state.g) ≈ 0.01*norm(f_g)
        # Mixed equalities and inequalities
        constraints = TwiceDifferentiableConstraints(
            (c,x)->(c[1]=x[1]*x[2]; c[2]=3*x[3]+x[4]^2),
            (J,x)->(J .= [x[2] x[1] 0 0; 0 0 3 2*x[4]]),
            (h,x,λ)->(h[4,4] += λ[2]*2),
            [], [], [0.1, 0.4], [0.1, 4.4])
        @test Optim.isfeasible(constraints, x)
        state = Optim.initial_state(method, options, d, constraints, x)
        Optim.update_fg!(d, constraints, state, method)
        J = zeros(2,4)
        constraints.jacobian!(J, x)
        eqnormal = vec(J[1,:]); eqnormal = eqnormal/norm(eqnormal)
        @test abs(dot(state.g, eqnormal)) < 1e-12  # orthogonal to equality constraint
        Pfg = f_g - dot(f_g, eqnormal)*eqnormal
        Pg = state.g - dot(state.g, eqnormal)*eqnormal
        @test norm(Pfg - Pg) ≈ 0.01*norm(Pfg)
        ## An objective function with a nonzero hessian
        hd = [1.0, 100.0, 0.01, 2.0]   # diagonal terms of hessian
        d = TwiceDifferentiable(x->sum(hd.*x.^2)/2, (g,x)->copyto!(g, hd.*x), (h,x)->copyto!(h, Diagonal(hd)), x)
        NLSolversBase.gradient!(d, x)
        gx = NLSolversBase.gradient(d)
        hx = Diagonal(hd)
        # Variable bounds
        constraints = TwiceDifferentiableConstraints([0.5, 0.0, -Inf, -Inf], [Inf, Inf, 1.0, 0.8])
        state = Optim.initial_state(method, options, d, constraints, x)
        Optim.update_fg!(d, constraints, state, method)
        @test abs(dot(gx, state.g)/dot(gx,gx) - 1) <= 0.011
        Optim.update_h!(d, constraints, state, method)
        @test abs(dot(gx, state.H*gx)/dot(gx, hx*gx) - 1) <= 0.011
        # Nonlinear inequalities
        constraints = TwiceDifferentiableConstraints(
            (c,x)->(c[1]=x[1]*x[2]; c[2]=3*x[3]+x[4]^2),
            (J,x)->(J[:,:] = [x[2] x[1] 0 0; 0 0 3 2*x[4]]),
            (h,x,λ)->(h[4,4] += λ[2]*2),
            [], [], [0.05, 0.4], [0.15, 4.4])
        @test Optim.isinterior(constraints, x)
        state = Optim.initial_state(method, options, d, constraints, x)
        Optim.update_fg!(d, constraints, state, method)
        @test abs(dot(gx, state.g)/dot(gx,gx) - 1) <= 0.011
        Optim.update_h!(d, constraints, state, method)
        @test abs(dot(gx, state.H*gx)/dot(gx, hx*gx) - 1) <= 0.011
        # Mixed equalities and inequalities
        constraints = TwiceDifferentiableConstraints(
            (c,x)->(c[1]=x[1]*x[2]; c[2]=3*x[3]+x[4]^2),
            (J,x)->(J[:,:] = [x[2] x[1] 0 0; 0 0 3 2*x[4]]),
            (h,x,λ)->(h[4,4] += λ[2]*2),
            [], [], [0.1, 0.4], [0.1, 4.4])
        @test Optim.isfeasible(constraints, x)
        state = Optim.initial_state(method, options, d, constraints, x)
        Optim.update_fg!(d, constraints, state, method)
        J = zeros(2,4)
        constraints.jacobian!(J, x)
        eqnormal = vec(J[1,:]); eqnormal = eqnormal/norm(eqnormal)
        @test abs(dot(state.g, eqnormal)) < 1e-12  # orthogonal to equality constraint
        Pgx = gx - dot(gx, eqnormal)*eqnormal
        @test abs(dot(Pgx, state.g)/dot(Pgx,Pgx) - 1) <= 0.011
        Optim.update_h!(d, constraints, state, method)
        @test abs(dot(Pgx, state.H*Pgx)/dot(Pgx, hx*Pgx) - 1) <= 0.011
    end

    @testset "IPNewton step" begin
        function autoqp(d, constraints, state)
            # Note that state must be fully up-to-date, and you must
            # have also called Optim.solve_step!
            p = Optim.pack_vec(state.x, state.bstate)
            chunksize = 1 #min(8, length(p))

            # TODO: How do we deal with the new Tags in ForwardDiff?

            # TD = ForwardDiff.Dual{chunksize, eltype(y)}
            TD = ForwardDiff.Dual{ForwardDiff.Tag{Nothing,Float64}, eltype(p), chunksize}

            # TODO: It doesn't seem like it is possible to to create a dual where the values are duals?
            # TD2 = ForwardDiff.Dual{chunksize, ForwardDiff.Dual{chunksize, eltype(p)}}
            # TD2 = ForwardDiff.Dual{ForwardDiff.Tag{Nothing,Float64}, typeof(TD), chunksize}
            Tx = typeof(state.x)
            stated = convert(Optim.IPNewtonState{TD, Tx,1}, state)
            # TODO: Uncomment
            #stated2 = convert(Optim.IPNewtonState{TD2, Tx, 1}, state)

            ϕd = αs->Optim.lagrangian_linefunc(αs, d, constraints, stated)
            # TODO: Uncomment
            #ϕd2 = αs->Optim.lagrangian_linefunc(αs, d, constraints, stated2)

            #ForwardDiff.gradient(ϕd, zeros(4)), ForwardDiff.hessian(ϕd2, zeros(4))
            ForwardDiff.gradient(ϕd, [0.0])#, ForwardDiff.hessian(ϕd2, [0.0])
        end
        F = 1000
        d = TwiceDifferentiable(x->F*x[1], (g, x) -> (g[1] = F), (h, x) -> (h[1,1] = 0), [0.0,])
        μ = 1e-20
        method = Optim.IPNewton(μ0=μ)
        options = Optim.Options(; Optim.default_options(method)...)
        x0 = μ/F*10  # minimum is at μ/F
        # Nonnegativity (the case that doesn't require slack variables)
        constraints = TwiceDifferentiableConstraints([0.0], [])
        state = Optim.initial_state(method, options, d, constraints, [x0])
        qp = Optim.solve_step!(state, constraints, options)
        @test state.s[1] ≈ -(F-μ/x0)/(state.bstate.λx[1]/x0)
        # TODO: Fix ForwardDiff
        #g0, H0 = autoqp(d, constraints, state)

        @test qp[1] ≈ F*x0-μ*log(x0)
        # TODO: Fix ForwardDiff
        #@test [qp[2]] ≈ g0 #-(F-μ/x0)^2*x0^2/μ
        # TODO: Fix ForwardDiff
        #@test [qp[3]] ≈ H0 # μ/x0^2*(x0 - F*x0^2/μ)^2
        bstate, bstep, bounds = state.bstate, state.bstep, constraints.bounds
        αmax = Optim.estimate_maxstep(Inf, state.x[bounds.ineqx].*bounds.σx,
                                          state.s[bounds.ineqx].*bounds.σx)
        ϕ = Optim.linesearch_anon(d, constraints, state, method)
        val0 = ϕ(0.0)
        val0 = isa(val0, Tuple) ? val0[1] : val0
        @test val0 ≈ qp[1]
        α = method.linesearch!(ϕ, 1.0, αmax, qp)
        @test α > 1e-3

        # TODO: Add linesearch_anon tests for IPNewton(linesearch! = backtracking_constrained)
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
        μ = 1e-20   # smaller than eps(1.0)
        method = Optim.IPNewton(μ0=μ)
        options = Optim.Options(; Optim.default_options(method)...)
        for σ in (1, -1)
            F = σ*F0
            # Nonnegativity/nonpositivity (the case that doesn't require slack variables)
            d = TwiceDifferentiable(x->F*x[1], (g, x) -> (g[1] = F), (h, x) -> (h[1,1] = 0), [0.0,])
            constraints = TwiceDifferentiableConstraints(σswap(σ, [0.0], [])...)
            state = Optim.initial_state(method, options, d, constraints, [μ/F*10])
            for i = 1:10
                Optim.update_state!(d, constraints, state, method, options)
                state.μ = μ
                Optim.update_fg!(d, constraints, state, method)
                Optim.update_h!(d, constraints, state, method)
            end
            @test isapprox(first(state.x), μ/F, rtol=1e-4)
            # |x| ≥ 1, and check that we get slack precision better than eps(1.0)
            d = TwiceDifferentiable(x->F*(x[1]-σ), (g, x) -> (g[1] = F), (h, x) -> (h[1,1] = 0), [0.0,])
            constraints = TwiceDifferentiableConstraints(σswap(σ, [Float64(σ)], [])...)
            state = Optim.initial_state(method, options, d, constraints, [(1+eps(1.0))*σ])
            for i = 1:10
                Optim.update_state!(d, constraints, state, method, options)
                state.μ = μ
                Optim.update_fg!(d, constraints, state, method)
                Optim.update_h!(d, constraints, state, method)
            end
            @test state.x[1] == σ
            @test state.bstate.slack_x[1] < eps(float(σ))
            # |x| >= 1 using the linear/nonlinear constraints
            d = TwiceDifferentiable(x->F*(x[1]-σ), (g, x) -> (g[1] = F), (h, x) -> (h[1,1] = 0), [0.0,])
            constraints = TwiceDifferentiableConstraints(
                (c,x)->(c[1] = x[1]),
                (J,x)->(J[1,1] = 1.0),
                (h,x,λ)->nothing,
                [], [], σswap(σ, [Float64(σ)], [])...)
            state = Optim.initial_state(method, options, d, constraints, [(1+eps(1.0))*σ])
            for i = 1:10
                Optim.update_state!(d, constraints, state, method, options)
                Optim.update_fg!(d, constraints, state, method)
                Optim.update_h!(d, constraints, state, method)
            end
            @test state.x[1] ≈ σ
            @test state.bstate.slack_c[1] < eps(float(σ))
        end
    end

    @testset "Constrained optimization" begin
        # TODO: Add more problems
        mcvp = MVP.ConstrainedProblems.examples
        method = IPNewton()

        for (name, prob) in mcvp
            debug_printing && printstyled("Problem: ", name, "\n", color=:green)
            df = TwiceDifferentiable(MVP.objective(prob), MVP.gradient(prob),
                                     MVP.objective_gradient(prob), MVP.hessian(prob), prob.initial_x)

            cd = prob.constraintdata
            constraints = TwiceDifferentiableConstraints(
                cd.c!, cd.jacobian!, cd.h!,
                cd.lx, cd.ux, cd.lc, cd.uc)

            options = Optim.Options(; Optim.default_options(method)...)

            minval = NLSolversBase.value(df, prob.solutions)

            results = optimize(df,constraints, prob.initial_x, method, options)
            @test isa(summary(results), String)
            @test Optim.converged(results)
            @test Optim.minimum(results) < minval + sqrt(eps(minval))

            debug_printing && printstyled("Iterations: $(Optim.iterations(results))\n", color=:red)
            debug_printing && printstyled("f-calls: $(Optim.f_calls(results))\n", color=:red)
            debug_printing && printstyled("g-calls: $(Optim.g_calls(results))\n", color=:red)
            debug_printing && printstyled("h-calls: $(Optim.h_calls(results))\n", color=:red)
        end

        # Test constraints on both x and c
        prob = mcvp["HS9"]
        df = TwiceDifferentiable(MVP.objective(prob), MVP.gradient(prob),
                                 MVP.objective_gradient(prob), MVP.hessian(prob), prob.initial_x)

        cd = prob.constraintdata
        lx = [5,5]; ux = [15,15]
        constraints = TwiceDifferentiableConstraints(
            cd.c!, cd.jacobian!, cd.h!,
            lx, ux, cd.lc, cd.uc)

        options = Optim.Options(; Optim.default_options(method)...)

        xsol = [9.,12.]
        x0   = [12. 14.0]
        minval = NLSolversBase.value(df, xsol)

        results = optimize(df,constraints, [12, 14.0], method, options)
        @test isa(Optim.summary(results), String)
        @test Optim.converged(results)
        @test Optim.minimum(results) < minval + sqrt(eps(minval))

        # Test the tracing
        @suppress_out begin
            # TODO: Update this when show_linesearch becomes part of Optim.Options
            method = IPNewton(show_linesearch = true)
            options = Optim.Options(iterations = 2,
                              show_trace = true, extended_trace=true, store_trace = true;
                              Optim.default_options(method)...)
            results = optimize(df,constraints, [12, 14.0], method, options)

            io = IOBuffer()
            show(io, results.trace)
            @test startswith(String(take!(io)), "Iter     Lagrangian value Function value   Gradient norm    |==constr.|      μ\n------   ---------------- --------------   --------------   --------------   --------\n")
        end

        # TODO: Add test where we start with an infeasible point
    end
end
