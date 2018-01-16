#= TODO:
- Support complex numbers
- Support manifolds
- How to deal with f_increased (as state and preconstate shares the same x and x_previous vectors)
- Check whether this makes sense for other preconditioners than GradientDescent and L-BFGS
* There might be some issue of dealing with x_current and x_previous in MomentumGradientDescent
* Trust region based methods may not work because we assume the preconditioner calls perform_linesearch!
=#

"""
Krylov subspace-type acceleration for optimization, N-GMRES.
Originally developed for solving nonlinear systems~\cite{washio1997krylov}, and reduces to
GMRES for linear problems.

A slight tweak that accelerates against the objective, instead of the norm of the gradient, is also implemented as O-ACCEL~\cite{riseth2017objective}.

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

@article{riseth2017objective,
   author = {Riseth, Asbj{\o}rn N.},
    title = "{Objective acceleration for unconstrained optimization}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1710.05200},
 primaryClass = "math.OC",
 keywords = {Mathematics - Optimization and Control, Mathematics - Numerical Analysis, 49M05, 65B99, 65K10},
     year = 2017,
    month = oct,
   adsurl = {http://adsabs.harvard.edu/abs/2017arXiv171005200N},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""

abstract type AbstractNGMRES <: FirstOrderOptimizer end

# TODO: Enforce TPrec <: Union{FirstOrderoptimizer,SecondOrderOptimizer}?
immutable NGMRES{IL, Tp,TPrec <: AbstractOptimizer,L} <: AbstractNGMRES
    alphaguess!::IL       # Initial step length guess for linesearch along direction xP->xA
    linesearch!::L        # Preconditioner moving from xP to xA (precondition x to accelerated x)
    nlprecon::TPrec       # Nonlinear preconditioner
    nlpreconopts::Options # Preconditioner options
    ϵ0::Tp                # Ensure A-matrix is positive definite
    wmax::Int             # Maximum window size
    # TODO: Add manifold support?
end

immutable OACCEL{IL, Tp,TPrec <: AbstractOptimizer,L} <: AbstractNGMRES
    alphaguess!::IL       # Initial step length guess for linesearch along direction xP->xA
    linesearch!::L        # Linesearch between xP and xA (precondition x to accelerated x)
    nlprecon::TPrec       # Nonlinear preconditioner
    nlpreconopts::Options # Preconditioner options
    ϵ0::Tp                # Ensure A-matrix is positive definite
    wmax::Int             # Maximum window size
    # TODO: Add manifold support?
end


Base.summary(s::NGMRES) = "Nonlinear GMRES preconditioned with $(summary(s.nlprecon))"
Base.summary(s::OACCEL) = "O-ACCEL preconditioned with $(summary(s.nlprecon))"

function NGMRES(;
                alphaguess = LineSearches.InitialStatic(),
                linesearch = LineSearches.HagerZhang(),
                nlprecon = GradientDescent(alphaguess = LineSearches.InitialPrevious(),
                                           linesearch = LineSearches.Static(alpha=1e-4,scaled=true)), # Step length arbitrary
                nlpreconopts = Options(iterations = 1, allow_f_increases = true),
                ϵ0 = 1e-12, # ϵ0 = 1e-12  -- number was an arbitrary choice
                wmax::Int = 10) # wmax = 10  -- number was an arbitrary choice to match L-BFGS field `m`
    # TODO: make wmax mandatory?
    NGMRES(alphaguess, linesearch, nlprecon, nlpreconopts, ϵ0, wmax)
end

function OACCEL(;
                alphaguess = LineSearches.InitialStatic(),
                linesearch = LineSearches.MoreThuente(),
                nlprecon = GradientDescent(alphaguess = LineSearches.InitialPrevious(),
                                           linesearch = LineSearches.Static(alpha=1e-4,scaled=true)), # Step length arbitrary
                nlpreconopts = Options(iterations = 1, allow_f_increases = true),
                ϵ0 = 1e-12, # ϵ0 = 1e-12  -- number was an arbitrary choice
                wmax::Int = 10) # wmax = 10  -- number was an arbitrary choice to match L-BFGS field `m`
    OACCEL(alphaguess, linesearch, nlprecon, nlpreconopts, ϵ0, wmax)
end


mutable struct NGMRESState{P,TA,T,N} <: AbstractOptimizerState where P <: AbstractOptimizerState
    # TODO: maybe we can just use nlpreconstate for x, x_previous and f_x_previous?
    x::Vector{T}              # Reference to nlpreconstate.x
    x_previous::Vector{T}     # Reference to nlpreconstate.x_previous
    x_previous_0::Vector{T}   # Used to deal with assess_convergence of NGMRES
    f_x_previous::T
    f_x_previous_0::T         # Used to deal with assess_convergence of NGMRES
    f_xP::T                   # For tracing purposes
    grnorm_xP::T              # For tracing purposes
    s::TA                     # Search direction for linesearch between xP and xA
    nlpreconstate::P          # Nonlinear preconditioner state
    X::Array{T,2}             # Solution vectors in the window (TODO: is this the best type?)
    R::Array{T,2}             # Gradient vectors in the window (TODO: is this the best type?)
    Q::Array{T,2}             # Storage to create linear system (TODO: make Symmetric?)
    ξ::Array{T}               # Storage to create linear system
    curw::Int                 # Counter for current window size
    A::Array{T,2}             # Container for Aα = b  (TODO: correct type?)
    b::Array{T,1}             # Container for Aα = b
    xA::Array{T,1}            # Container for accelerated step
    k::Int                    # Used for indexing where to put values in the Storage containers
    restart::Bool             # Restart flag
    g_tol::T                  # Exit tolerance to be checked after nonlinear preconditioner apply
    subspacealpha::Array{T,1} # Storage for coefficients in the subspace for the acceleration step
    @add_linesearch_fields()
end

"Update storage Q[i,j] and Q[j,i] for `NGMRES`"
@inline function _updateQ!(Q, i::Int, j::Int, X, R, ::NGMRES)
    Q[j,i] = dot(R[:, j], R[:,i]) # TODO: vecdot?
    if i != j
        Q[i,j] = Q[j, i]          # TODO: Use Symmetric?
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

"Update value η for `NGMRES`"
@inline function _updateη(x, r, ::NGMRES)
    dot(r, r) # TODO: vecdot?
end

"Update storage Q[i,j] and Q[j,i] for `OACCEL`"
@inline function _updateQ!(Q, i::Int, j::Int, X, R, ::OACCEL)
    Q[i,j] = dot(X[:,i], R[:,j])     #TODO: vecdot?
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

"Update value η for `OACCEL`"
@inline function _updateη(x, r, ::OACCEL)
    dot(x, r) # TODO: vecdot?
end


function initial_state(method::AbstractNGMRES, options, d, initial_x::AbstractArray{T}) where T
    if !(typeof(method.nlprecon) <: Union{GradientDescent,LBFGS})
        Base.warn_once("Use caution. N-GMRES/O-ACCEL has only been tested with Gradient Descent and L-BFGS preconditioning.")
    end
    nlpreconstate = initial_state(method.nlprecon, method.nlpreconopts, d, initial_x)
    wmax = method.wmax

    X = Array{T}(length(initial_x), wmax)
    R = Array{T}(length(initial_x), wmax)
    Q = Array{T}(wmax, wmax)

    ξ = if typeof(method) <: OACCEL
        Array{T}(wmax, 2)
    else
        Array{T}(wmax)
    end

    copy!(view(X,:,1), initial_x)
    copy!(view(R,:,1), gradient(d))

    _updateQ!(Q, 1, 1, X, R, method)

    NGMRESState(nlpreconstate.x,          # Maintain current state in state.x. Use same vector as preconditioner.
                nlpreconstate.x_previous, # Maintain  in state.x_previous. Use same vector as preconditioner.
                similar(nlpreconstate.x), # Maintain state at the beginning of an iteration in state.x_previous_0. Used for convergence asessment.
                T(NaN),                   # Store previous f in state.f_x_previous
                T(NaN),                   # Store f value from the beginning of an iteration in state.f_x_previous_0. Used for convergence asessment.
                T(NaN),                   # Store value f_xP of f(x^P) for tracing purposes
                T(NaN),                   # Store value grnorm_xP of |g(x^P)| for tracing purposes
                similar(initial_x),       # Maintain current search direction in state.s
                nlpreconstate,            # State storage for preconditioner
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
                options.g_tol,            # Exit tolerance check after nonlinear preconditioner apply
                Array{T}(wmax),           # subspacealpha
                @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

nlprecon_post_optimize!(d, state, method) = update_h!(d, state.nlpreconstate, method)

function nlprecon_post_optimize!(d, state::NGMRESState{X,T},
                                 method::Union{LBFGS,BFGS}) where X where T
    update_h!(d, state.nlpreconstate, method)
end

nlprecon_post_accelerate!(d, state, method) = update_h!(d, state.nlpreconstate, method)

function nlprecon_post_accelerate!(d, state::NGMRESState{X,T},
                                   method::Union{LBFGS,BFGS})  where X where T
    state.nlpreconstate.pseudo_iteration += 1
    update_h!(d, state.nlpreconstate, method)
end


function update_state!(d, state::NGMRESState{X,T}, method::AbstractNGMRES) where X where T
    # Maintain a record of previous position, for convergence assessment
    copy!(state.x_previous_0, state.x)
    state.f_x_previous_0 = value(d)

    state.k += 1
    curw = state.curw

    # Step 1: Call preconditioner to get x^P
    res = optimize(d, state.x, method.nlprecon, method.nlpreconopts, state.nlpreconstate)

    if any(.!isfinite.(state.x)) || any(.!isfinite.(gradient(d))) || !isfinite(value(d))
        warn("Non-finite values attained from preconditioner $(summary(method.nlprecon)).")
        return true
    end


    # Calling value_gradient! in normally done on state.x in optimize or update_g! above,
    # but there are corner cases where we need this.
    state.f_xP = value_gradient!(d, state.x)
    gP = gradient(d)
    state.grnorm_xP = norm(gP, Inf)

    if g_residual(gP) ≤ state.g_tol
        return true # Exit on gradient norm convergence
    end

    # Deals with update_h! etc for preconditioner, if needed
    nlprecon_post_optimize!(d, state, method.nlprecon)

    # Step 2: Do acceleration calculation
    η = _updateη(state.x, gP, method)

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
        # TODO: set the restart flag?
        Base.warn("Calculated α is NaN in $(summary(method))")
        state.s .= zero(eltype(state.s))
    else
        # xA = xP + \sum_{j=1}^{curw} α[j] * (X[j] - xP)
        state.xA .= (1.0-sum(α)).*state.x .+ (state.X[:,1:curw] * α) # TODO: less alloc with sum?
        @. state.s = state.xA - state.x
    end

    # 3: Perform condition checks
    if dot(state.s, gP) ≥ 0 || !isfinite(dot(state.s, gP))
        # Moving from xP to xA is *not* a descent direction
        # Discard xA
        state.restart = true # TODO: expand restart heuristics
        lssuccess = true
        state.alpha = 0.0
    else
        state.restart = false

        # Update f_x_previous and dphi0_previous according to preconditioner step
        # This may be used in perform_linesearch!/alphaguess! when moving from x^P to x^A
        # TODO: make this a function?
        state.f_x_previous = state.nlpreconstate.f_x_previous
        # TODO: Use dphi0_previous when alphaguess branch is merged
        #state.dphi0_previous = state.nlpreconstate.dphi0_previous # assumes precon is a linesearch based method. TODO: Deal with trust region based methods
        # state.x_previous and state.x are dealt with by reference

        lssuccess = perform_linesearch!(state, method, d)
        @. state.x = state.x + state.alpha * state.s

        # TODO: Move these into `nlprecon_post_accelerate!` ?
        state.nlpreconstate.f_x_previous = state.f_x_previous
        state.nlpreconstate.dphi0_previous = state.dphi0_previous
        # Deals with update_h! etc. for preconditioner, if needed
        nlprecon_post_accelerate!(d, state, method.nlprecon)
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
