"""
Krylov subspace-type acceleration for optimization.
Originally developed for solving nonlinear systems~\cite{washio1997krylov}, and reduces to
GMRES for linear problems.

Application of the algorithm to optimization is covered, for example, in~\cite{sterck2013steepest}

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

immutable NGMRES{Tp,TPrec <: Optimizer,L} <: Optimizer
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

Base.summary(s::NGMRES) = "Nonlinear GMRES preconditioned with $(summary(s.precon))"

function NGMRES(; linesearch = LineSearches.Static(),
                precon = GradientDescent(linesearch = LineSearches.Static()),
                preconopts = Options(iterations = 1),
                # γA = 2.0, γB = 0.9, # (defaults in Washio and Oosterlee)
                # γC = 2.0, ϵB = 0.1, # (defaults in Washio and Oosterlee)
                ϵ0 = 1e-12, # ϵ0 = 1e-12  -- number was an arbitrary choice
                wmax = 10) # wmax = 10  -- number was an arbitrary choice
    # TODO: make wmax mandatory?
    NGMRES(linesearch, precon, preconopts, #γA, γB, γC, ϵB,
           ϵ0, wmax)
end

mutable struct NGMRESState{P,TA,T,N}
    # TODO: maybe we can just use preconstate for x, x_previous and f_x_previous?
    x::Vector{T} # TODO: How to handle non-vector types?
    x_previous::Vector{T}
    f_x_previous::T
    s::TA # Search direction for linesearch between xP and xA
    # TODO: Specify preconstate::P where P <: AbstractOptimizerState
    preconstate::P  # Preconditioner state
    X::Array{T,2}  # Solution vectors in the window (TODO: is this the best type?)
    R::Array{T,2}  # Gradient vectors in the window (TODO: is this the best type?)
    Q::Array{T,2}   # Storage to create linear system (TODO: make Symmetric?)
    ξ::Array{T,1}   # Storage to create linear system
    curw::Int       # Counter for current window size
    A::Array{T,2}   # Container for Aα = b  (TODO: correct type?)
    b::Array{T,1}   # Container for Aα = b
    xA::Array{T,1}  # Container for accelerated step
    iteration::Int  # Iteration counter
    restart::Bool   # Restart flag
    @add_linesearch_fields()
end

function initial_state(method::NGMRES, options, d, initial_x::AbstractArray{T}) where T
    preconstate = initial_state(method.precon, method.preconopts, d, initial_x)
    wmax = method.wmax

    X = Array{T}(length(initial_x), wmax)
    R = Array{T}(length(initial_x), wmax)
    Q = Array{T}(wmax, wmax)

    copy!(X[:,1], initial_x)
    copy!(R[:,1], gradient(d))
    Q[1,1] =  dot(gradient(d), gradient(d))

    NGMRESState(initial_x,                      # Maintain current state in state.x
                similar(initial_x),             # Maintain previous state in state.x_previous
                T(NaN),                         # Store previous f in state.f_x_previous
                similar(initial_x),             # Maintain current search direction in state.s
                preconstate,                    # State storage for preconditioner
                X,
                R,
                Q,
                Array{T}(wmax),                 # ξ
                1,                              # curw
                Array{T}(wmax, wmax),           # A
                Array{T}(wmax),                 # b
                similar(initial_x),             # xA
                0,                              # iteration counter
                false,
                @initial_linesearch()...)       # Maintain a cache for line search results in state.lsr
end

function update_state!(d, state::NGMRESState{X,T}, method::NGMRES) where X where T # Do I need to specify X, T here?
    # Maintain a record of previous position
    copy!(state.x_previous, state.x)
    state.f_x_previous  = value(d)

    state.iteration += 1
    curw = state.curw

    # Step 1: Call preconditioner to get x^P
    res = optimize(d, state.x, method.precon, method.preconopts, state.preconstate)

    # state.x = xP
    state.x .= Optim.minimizer(res)

    # TODO: check for convergence
    # assess_convergence(with xP and team (so using preconstate?)


    # Step 2: do NGMRES minimization
    gradient!(d, state.x)
    gP = gradient(d)
    η = dot(gP, gP)

    for i = 1:curw
        state.ξ[i] = dot(gP, state.R[:,i])
        state.b[i] = η-state.ξ[i]
    end

    for i = 1:curw
        for j = 1:curw
            state.A[i,j] = state.Q[i,j]-state.ξ[i]-state.ξ[j]+η;
        end
    end

    # The outer max is to avoid δ=0, which may occur if A=0, e.g. at numerical convergence
    δ = method.ϵ0*max(maximum(diag(state.A)[1:curw]), method.ϵ0)

    # TODO: preallocate α in state?
    #state.α[1:curw] .= (state.A[1:curw,1:curw] + δ*I) \ state.b[1:curw]
    α = (state.A[1:curw,1:curw] + δ*I) \ state.b[1:curw]

    if any(isnan, α)
        # TODO: set the restart flag
        Base.warn("Calculated α is NaN in N-GMRES.")
    end

    # xA = xP + \sum_{j=1}^{curw} α[j] * (X[j] - xP)
    state.xA .= (1.0-sum(α)).*state.x .+ (state.X[:,1:curw] * α) # TODO: better way?

    # 3: Perform condition checks
    @. state.s = state.xA - state.x

    if dot(state.s, gP) ≥ 0
        # Moving from xP to xA is *not* a descent direction
        # Discard xA
        state.restart = true # TODO: expand restart heuristics
    else
        state.restart = false
        perform_linesearch!(state, method, d)
    end

    false
end

function update_g!(d, state, method::NGMRES)
    # Update the function value and gradient
    value_gradient!(d, state.x)
    if state.restart == false
        state.curw = max(state.curw + 1, method.wmax)
        j = mod(state.iteration, method.wmax) + 1
    else
        state.restart = true
        state.curw = 1
        j = 1
    end

    copy!(state.X[:,j], state.x)
    copy!(state.R[:,j], gradient(d))
    for i=1:state.curw
        state.Q[j,i] = dot(gradient(d), state.R[:,i])
        state.Q[i,j] = state.Q[j,i] # Use Symmetric?
    end
end

function trace!(tr, d, state, iteration, method::NGMRES, options)
    # TODO: create NGMRES-specific trace
    Base.warn_once("We need to implement proper tracing for N-GMRES")
    common_trace!(tr, d, state, iteration, method, options)
end

function assess_convergence(state::NGMRESState, d, options)
  default_convergence_assessment(state, d, options)
end
