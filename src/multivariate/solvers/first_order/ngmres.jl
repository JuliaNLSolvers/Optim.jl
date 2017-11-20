#= TODO:
- Support complex numbers
- Support manifolds
- How to deal with f_increased (as state and preconstate shares the same x and x_previous vectors)
- Check whether this makes sense for other preconditioners than GradientDescent
  * Update g_previous for L-BFGS after the accelerated step is taken
  * There might be some issue of dealing with x_current and x_previous in MomentumGradientDescent
  * Trust region based methods won't work because we assume the preconditioner calls perform_linesearch!
=#

"""
Krylov subspace-type acceleration for optimization.
Originally developed for solving nonlinear systems~\cite{washio1997krylov}, and reduces to
GMRES for linear problems.

Application of the algorithm to optimization is covered, for example, in~\cite{sterck2013steepest}.

@article{washio1997krylov,
  title={Krylov subspace acceleration for nonlinear multigrid schemes},
  author={Washio, T and Oosterlee, CW},
  journal={Electronic Transactions on Numerical Analysis},
  volume={6},
  number={271-290},
  pages={3--1},
  year={1997}
}

@article{sterck2013steepest,
  title={Steepest descent preconditioning for nonlinear GMRES optimization},
  author={Sterck, Hans De},
  journal={Numerical Linear Algebra with Applications},
  volume={20},
  number={3},
  pages={453--471},
  year={2013},
  publisher={Wiley Online Library}
}
"""

abstract type AbstractNGMRES <: FirstOrderOptimizer end

immutable NGMRES{IL, Tp,TPrec <: Optimizer,L} <: AbstractNGMRES
    alphaguess!::IL   # Initial step length guess for linesearch along direction xP->xA
    linesearch!::L    # Preconditioner moving from xP to xA (precondition x to accelerated x)
    precon::TPrec # Nonlinear preconditioner
    preconopts::Options # Preconditioner options
    #γA::Tc # Heuristic condition (Washio and Oosterlee)
    #γB::Tc # Heuristic condition (Washio and Oosterlee)
    #ϵB::Tc # Heuristic condition (Washio and Oosterlee)
    #γC::Tc # Heuristic condition (Washio and Oosterlee)
    ϵ0::Tp # Ensure A-matrix is positive definite
    wmax::Int # Maximum window size
    # TODO: Add manifold support?
end

immutable OACCEL{IL, Tp,TPrec <: Optimizer,L} <: AbstractNGMRES
    alphaguess!::IL   # Initial step length guess for linesearch along direction xP->xA
    linesearch!::L    # Linesearch between xP and xA (precondition x to accelerated x)
    precon::TPrec # Nonlinear preconditioner
    preconopts::Options # Preconditioner options
    ϵ0::Tp # Ensure A-matrix is positive definite
    wmax::Int # Maximum window size
    # TODO: Add manifold support?
end


Base.summary(s::NGMRES) = "Nonlinear GMRES preconditioned with $(summary(s.precon))"
Base.summary(s::OACCEL) = "O-ACCEL preconditioned with $(summary(s.precon))"

function NGMRES(;
                alphaguess = LineSearches.InitialStatic(),
                linesearch = LineSearches.MoreThuente(),
                precon = GradientDescent(alphaguess = LineSearches.InitialPrevious(),
                                         linesearch = LineSearches.Static(alpha=1e-4,scaled=true)), # Step length arbitrary
                preconopts = Options(iterations = 1, allow_f_increases = true),
                # γA = 2.0, γB = 0.9, # (defaults in Washio and Oosterlee)
                # γC = 2.0, ϵB = 0.1, # (defaults in Washio and Oosterlee)
                ϵ0 = 1e-12, # ϵ0 = 1e-12  -- number was an arbitrary choice
                wmax::Int = 10) # wmax = 10  -- number was an arbitrary choice
    # TODO: make wmax mandatory?
    NGMRES(alphaguess, linesearch, precon, preconopts, #γA, γB, γC, ϵB,
           ϵ0, wmax)
end

function OACCEL(;
                alphaguess = LineSearches.InitialStatic(),
                linesearch = LineSearches.MoreThuente(),
                precon = GradientDescent(alphaguess = LineSearches.InitialPrevious(),
                                         linesearch = LineSearches.Static(alpha=1e-4,scaled=true)), # Step length arbitrary
                preconopts = Options(iterations = 1, allow_f_increases = true),
                ϵ0 = 1e-12, # ϵ0 = 1e-12  -- number was an arbitrary choice
                wmax::Int = 10) # wmax = 10  -- number was an arbitrary choice
    OACCEL(alphaguess, linesearch, precon, preconopts, ϵ0, wmax)
end


mutable struct NGMRESState{P,TA,T,N} #where P <: AbstractOptimizerState
    # TODO: maybe we can just use preconstate for x, x_previous and f_x_previous?
    x::Vector{T} # Reference to preconstate.x
    x_previous::Vector{T}   # Reference to preconstate.x_previous
    x_previous_0::Vector{T} # Used to deal with assess_convergence of NGMRES
    f_x_previous::T
    f_x_previous_0::T # Used to deal with assess_convergence of NGMRES
    f_xP::T         # For tracing purposes
    grnorm_xP::T     # For tracing purposes
    s::TA # Search direction for linesearch between xP and xA
    # TODO: Specify preconstate::P where P <: AbstractOptimizerState
    preconstate::P  # Preconditioner state
    X::Array{T,2}  # Solution vectors in the window (TODO: is this the best type?)
    R::Array{T,2}  # Gradient vectors in the window (TODO: is this the best type?)
    Q::Array{T,2}   # Storage to create linear system (TODO: make Symmetric?)
    ξ::Array{T}     # Storage to create linear system
    curw::Int       # Counter for current window size
    A::Array{T,2}   # Container for Aα = b  (TODO: correct type?)
    b::Array{T,1}   # Container for Aα = b
    xA::Array{T,1}  # Container for accelerated step
    k::Int          # Used for indexing where to put values in the Storage containers
    restart::Bool   # Restart flag
    g_tol::T        # Exit tolerance to be checked after preconditioner apply
    subspacealpha::Array{T,1}  # Storage for coefficients in the subspace for the acceleration step
    @add_linesearch_fields()
end

"Update storage Q[i,j] and Q[j,i] for `NGMRES`"
@inline function _updateQ!(Q, i::Int, j::Int, X, R, ::NGMRES)
    Q[j,i] = dot(R[:, j], R[:,i]) #TODO: vecdot?
    if i != j
        Q[i,j] = Q[j, i] # Use Symmetric?
    end
end

"Update storage A[i,j] for `NGMRES`"
@inline function _updateA!(A, i::Int, j::Int, Q, ξ, η, ::NGMRES)
    A[i,j] = Q[i,j]-ξ[i]-ξ[j]+η
end

"Update storage ξ[i,:] for `NGMRES`"
@inline function _updateξ!(ξ, i::Int, X, x, R, r, ::NGMRES)
    ξ[i] = dot(r, R[:,i])
end

"Update storage b[i] for `NGMRES`"
@inline function _updateb!(b, i::Int, ξ, η, ::NGMRES)
    b[i] = η - ξ[i]
end

"Update storage Q[i,j] and Q[j,i] for `OACCEL`"
@inline function _updateQ!(Q, i::Int, j::Int, X, R, ::OACCEL)
    Q[i,j] = dot(X[:,i], R[:,j]) #TODO: vecdot?
    if i != j
        Q[j,i] = dot(X[:,j], R[:,i]) #TODO: vecdot?
    end
end

"Update storage A[i,j] for `OACCEL`"
@inline function _updateA!(A, i::Int, j::Int, Q, ξ, η, ::OACCEL)
    A[i,j] = Q[i,j]-ξ[i,1]-ξ[j,2]+η
end

"Update storage ξ[i,:] for `OACCEL`"
@inline function _updateξ!(ξ, i::Int, X, x, R, r, ::OACCEL)
    ξ[i,1] = dot(X[:,i], r)
    ξ[i,2] = dot(x, R[:,i])
end

"Update storage b[i] for `OACCEL`"
@inline function _updateb!(b, i::Int, ξ, η, ::OACCEL)
    b[i] = η - ξ[i,1]
end


function initial_state(method::AbstractNGMRES, options, d, initial_x::AbstractArray{T}) where T
    if !(typeof(method.precon) <: GradientDescent)
        warn_once("Use caution. NGMRES has only been tested with Gradient Descent preconditioning")
    end
    preconstate = initial_state(method.precon, method.preconopts, d, initial_x)
    wmax = method.wmax

    X = Array{T}(length(initial_x), wmax)
    R = Array{T}(length(initial_x), wmax)
    Q = Array{T}(wmax, wmax)

    ξ = if typeof(method) == OACCEL
        Array{T}(wmax, 2)
    else
        Array{T}(wmax)
    end

    copy!(view(X,:,1), initial_x)
    copy!(view(R,:,1), gradient(d))

    _updateQ!(Q, 1, 1, X, R, method)

    NGMRESState(preconstate.x,            # Maintain current state in state.x. Use same vector as preconditioner.
                preconstate.x_previous,   # Maintain  in state.x_previous. Use same vector as preconditioner.
                similar(preconstate.x),   # Maintain state at the beginning of an iteration in state.x_previous_0. Used for convergence asessment.
                T(NaN),                   # Store previous f in state.f_x_previous
                T(NaN),                   # Store f value from the beginning of an iteration in state.f_x_previous_0. Used for convergence asessment.
                T(NaN),                   # Store value f_xP of f(x^P) for tracing purposes
                T(NaN),                   # Store value grnorm_xP of |g(x^P)| for tracing purposes
                similar(initial_x),       # Maintain current search direction in state.s
                preconstate,              # State storage for preconditioner
                X,
                R,
                Q,
                ξ,
                1,                        # curw
                Array{T}(wmax, wmax),     # A
                Array{T}(wmax),           # b
                similar(initial_x),       # xA
                0,                        # iteration counter
                false,                    # Restart flag
                options.g_tol,            # Exit tolerance check after preconditioner apply
                Array{T}(wmax),           # subspacealpha
                @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end


function update_state!(d, state::NGMRESState{X,T}, method::AbstractNGMRES) where X where T

    # Reset step length for preconditioner, in case the previous value is detrimental
    state.preconstate.alpha = one(state.preconstate.alpha)

    # Maintain a record of previous position, for convergence assessment
    copy!(state.x_previous_0, state.x)
    state.f_x_previous_0 = value(d)

    state.k += 1
    curw = state.curw
    # Step 1: Call preconditioner to get x^P
    res = optimize(d, state.x, method.precon, method.preconopts, state.preconstate)

    if g_residual(gradient(d)) ≤ state.g_tol
        return false # Exit on gradient norm convergence
    end

    # Step 2: do NGMRES minimization
    state.f_xP = value_gradient!(d, state.x) # TODO: calling value_gradient! should be superflous, as the last evaluation of d should be at state.x
    gP = gradient(d)
    state.grnorm_xP = norm(gP, Inf)

    η = dot(gP, gP)

    for i = 1:curw
        # Update storage vectors according to method {NGMRES, OACCEL}
        _updateξ!(state.ξ, i, state.X, state.x, state.R, gP, method)
        _updateb!(state.b, i, state.ξ, η, method)
    end

    for i = 1:curw
        for j = 1:curw
            # Update system matrix according to method {NGMRES, OACCEL}
            _updateA!(state.A, i, j, state.Q, state.ξ, η, method)
        end
    end

    # The outer max is to avoid δ=0, which may occur if A=0, e.g. at numerical convergence
    δ = method.ϵ0*max(maximum(diag(state.A)[1:curw]), method.ϵ0)

    #state.α[1:curw] .= (state.A[1:curw,1:curw] + δ*I) \ state.b[1:curw]
    α = view(state.subspacealpha, 1:curw)
    α .= (state.A[1:curw,1:curw] + δ*I) \ state.b[1:curw]
    if any(isnan, α)
        # TODO: set the restart flag
        Base.warn("Calculated α is NaN in N-GMRES.")
        state.s .= zero(eltype(state.s))
    else
        # xA = xP + \sum_{j=1}^{curw} α[j] * (X[j] - xP)
        state.xA .= (1.0-sum(α)).*state.x .+ (state.X[:,1:curw] * α) # TODO: less alloc with sum?
        @. state.s = state.xA - state.x
    end

    # 3: Perform condition checks
    if dot(state.s, gP) ≥ 0
        # Moving from xP to xA is *not* a descent direction
        # Discard xA
        state.restart = true # TODO: expand restart heuristics
        lssuccess = true
    else
        state.restart = false

        # Update f_x_previous and dphi0_previous according to preconditioner step
        # This may be used in perform_linesearch!/alphaguess! when moving from x^P to x^A
        # TODO: make this a function?
        state.f_x_previous = state.preconstate.f_x_previous
        # TODO: Use dphi0_previous when alphaguess branch is merged
        #state.dphi0_previous = state.preconstate.dphi0_previous # assumes precon is a linesearch based method. TODO: Deal with trust region based methods
        # state.x_previous and state.x are dealt with by reference

        lssuccess = perform_linesearch!(state, method, d)
        @. state.x = state.x + state.alpha * state.s

        # TODO: Make this into a function
        state.preconstate.f_x_previous = state.f_x_previous
        # TODO: Use dphi0_previous when alphaguess branch is merged
        #state.preconstate.dphi0_previous = state.dphi0_previous
    end
    #=
    Update x_previous and f_x_previous to be the values at the beginning
    of the N-GMRES iteration. For convergence assessment purposes.
    =#
    copy!(state.x_previous, state.x_previous_0)
    state.f_x_previous = state.f_x_previous_0

    lssuccess == false # Break on linesearch error
end

function update_g!(d, state, method::AbstractNGMRES)
    # Update the function value and gradient
    value_gradient!(d, state.x)
    if state.restart == false
        state.curw = min(state.curw + 1, method.wmax)
    else
        state.k = 0
        state.curw = 1
    end
    j = mod(state.k, method.wmax) + 1

    copy!(view(state.X,:,j), state.x)
    copy!(view(state.R,:,j), gradient(d))

    for i = 1:state.curw
        _updateQ!(state.Q, i, j, state.X, state.R, method)
    end
end

function trace!(tr, d, state, iteration, method::AbstractNGMRES, options)
    dt = Dict()
    if state.restart == false

    end
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        if state.restart == true
            dt["Current step size"] = NaN
        else
            dt["Current step size"] = state.alpha
            # This is a wasteful hack to get the previous values for debugging purposes only.
            xP = state.x .- state.alpha .* state.s
            dt["x^P"] = copy(xP)
            # TODO: What's a good way to include g(x^P) here without messing up gradient counts?
        end
    end
    dt["Restart"] = state.restart
    if state.restart == false
        dt["f(x^P)"] = state.f_xP
        dt["|g(x^P)|"] = state.grnorm_xP
    end

    g_norm = vecnorm(gradient(d), Inf)
    update!(tr,
            iteration,
            value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end

function assess_convergence(state::NGMRESState, d, options)
    default_convergence_assessment(state, d, options)
end
