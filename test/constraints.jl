using Optim
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

    @testset "Lagrangian val/grad" begin
        function check_autodiff(d, bounds, x, cfun::Function, bstate, μ)
            c = cfun(x)
            J = ForwardDiff.jacobian(cfun, x)
            # Using real-valued inputs
            p = Optim.pack_vec(x, bstate)
            ftot! = (p,storage)->Optim.lagrangian_fgvec!(p, storage, gx, bgrad, d, bounds, x, c, J, bstate, μ, nothing)
            pgrad = similar(p)
            ftot!(p, pgrad)
            # Compute with ForwardDiff
            chunksize = min(8, length(p))
            TD = ForwardDiff.Dual{chunksize,eltype(p)}
            xd = Array{TD}(length(x))
            bstated = Optim.BarrierStateVars{TD}(bounds)
            pcmp = similar(p)
            ftot = p->Optim.lagrangian_vec(p, d, bounds, xd, cfun, bstated, μ, nothing)
            ForwardDiff.gradient!(pcmp, ftot, p, ForwardDiff.Chunk{chunksize}())
            @test pcmp ≈ pgrad
        end
        # Basic setup
        μ = 0.2345678
        A = randn(3,3); H = A'*A
        d = DifferentiableFunction(x->(x'*H*x)[1]/2, (x,storage)->(storage[:] = H*x))
        x = broadcast(clamp, randn(3), -0.99, 0.99)
        gx = similar(x)
        cfun = x->Float64[]
        c = Float64[]
        J = Array{Float64}(0,0)
        ## No constraints
        bounds = Optim.ConstraintBounds(Float64[], Float64[], Float64[], Float64[])
        bstate = Optim.BarrierStateVars(bounds, x)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, Float64[], Array{Float64}(0,0), bstate, μ, nothing)
        @test f_x == L == d.f(x)
        @test gx == H*x
        ## Pure equality constraints on variables
        d = DifferentiableFunction(x->0.0, (x,storage)->fill!(storage, 0))
        xbar = fill(0.2, length(x))
        bounds = Optim.ConstraintBounds(xbar, xbar, [], [])
        bstate = Optim.BarrierStateVars(bounds)
        rand!(bstate.λxE)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, c, J, bstate, μ, nothing)
        @test f_x == 0
        @test L ≈ dot(bstate.λxE, xbar-x)
        @test gx == -bstate.λxE
        @test bgrad.λxE == xbar-x
        check_autodiff(d, bounds, x, cfun, bstate, μ)
        ## Nonnegativity constraints
        bounds = Optim.ConstraintBounds(zeros(length(x)), fill(Inf,length(x)), [], [])
        y = rand(length(x))
        bstate = Optim.BarrierStateVars(bounds, y)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, y, Float64[], Array{Float64}(0,0), bstate, μ, nothing)
        @test f_x == 0
        @test L ≈ -μ*sum(log, y)
        @test gx == -μ./y
        check_autodiff(d, bounds, y, cfun, bstate, μ)
        ## General inequality constraints on variables
        bounds = Optim.ConstraintBounds(rand(length(x))-2, rand(length(x))+1, [], [])
        bstate = Optim.BarrierStateVars(bounds, x)
        rand!(bstate.slack_x)  # intentionally displace from the correct value
        rand!(bstate.λx)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, Float64[], Array{Float64}(0,0), bstate, μ, nothing)
        @test f_x == 0
        Ltarget = -μ*sum(log, bstate.slack_x) +
            dot(bstate.λx, bstate.slack_x - bounds.σx.*(x[bounds.ineqx]-bounds.bx))
        @test L ≈ Ltarget
        dx = similar(gx); fill!(dx, 0)
        for (i,j) in enumerate(bounds.ineqx)
            dx[j] -= bounds.σx[i]*bstate.λx[i]
        end
        @test gx ≈ dx
        @test bgrad.slack_x == -μ./bstate.slack_x + bstate.λx
        check_autodiff(d, bounds, x, cfun, bstate, μ)
        ## Nonlinear equality constraints
        cfun = x->[x[1]^2+x[2]^2, x[2]*x[3]^2]
        c = cfun(x)
        J = ForwardDiff.jacobian(cfun, x)
        cbar = rand(length(c))
        bounds = Optim.ConstraintBounds([], [], cbar, cbar)
        bstate = Optim.BarrierStateVars(bounds, x, c)
        rand!(bstate.λcE)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, c, J, bstate, μ, nothing)
        @test f_x == 0
        @test L ≈ dot(bstate.λcE, cbar-c)
        @test gx ≈ -J'*bstate.λcE
        @test bgrad.λcE == cbar-c
        check_autodiff(d, bounds, x, cfun, bstate, μ)
        ## Nonlinear inequality constraints
        bounds = Optim.ConstraintBounds([], [], rand(length(c))-1, rand(length(c))+1)
        bstate = Optim.BarrierStateVars(bounds, x, c)
        rand!(bstate.slack_c)  # intentionally displace from the correct value
        rand!(bstate.λc)
        bgrad = similar(bstate)
        f_x, L = Optim.lagrangian_fg!(gx, bgrad, d, bounds, x, c, J, bstate, μ, nothing)
        @test f_x == 0
        Ltarget = -μ*sum(log, bstate.slack_c) +
            dot(bstate.λc, bstate.slack_c - bounds.σc.*(c[bounds.ineqc]-bounds.bc))
        @test L ≈ Ltarget
        @test gx ≈ -J[bounds.ineqc,:]'*(bstate.λc.*bounds.σc)
        @test bgrad.slack_c == -μ./bstate.slack_c + bstate.λc
        check_autodiff(d, bounds, x, cfun, bstate, μ)
    end
end

nothing
