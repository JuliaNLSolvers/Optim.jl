struct IPNewton{F,Tμ<:Union{Symbol,Number}} <: IPOptimizer{F}
    linesearch!::F
    μ0::Tμ      # Initial value for the barrier penalty coefficient μ
    show_linesearch::Bool
    # TODO: μ0, and show_linesearch were originally in options
end

Base.summary(::IPNewton) = "Interior Point Newton"

# TODO: Add support for InitialGuess from LineSearches
"""
# Interior-point Newton
## Constructor
```jl
IPNewton(; linesearch::Function = Optim.backtrack_constrained_grad,
         μ0::Union{Symbol,Number} = :auto,
         show_linesearch::Bool = false)
```

The initial barrier penalty coefficient `μ0` can be chosen as a number, or set
to `:auto` to let the algorithm decide its value, see `initialize_μ_λ!`.

*Note*: For constrained optimization problems, we recommend
always enabling `allow_f_increases` and `successive_f_tol` in the options passed to `optimize`.
The default is set to `Optim.Options(allow_f_increases = true, successive_f_tol = 2)`.

As of February 2018, the line search algorithm is specialised for constrained
interior-point methods. In future we hope to support more algorithms from
`LineSearches.jl`.

## Description
The `IPNewton` method implements an interior-point primal-dual Newton algorithm for solving
nonlinear, constrained optimization problems. See Nocedal and Wright (Ch. 19, 2006) for a discussion of
interior-point methods for constrained optimization.

## References
The algorithm was [originally written by Tim Holy](https://github.com/JuliaNLSolvers/Optim.jl/pull/303) (@timholy, tim.holy@gmail.com).

 - J Nocedal, SJ Wright (2006), Numerical optimization, second edition. Springer.
 - A Wächter, LT Biegler (2006), On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Mathematical Programming 106 (1), 25-57.
"""
IPNewton(; linesearch::Function = backtrack_constrained_grad,
         μ0::Union{Symbol,Number} = :auto,
         show_linesearch::Bool = false) =
  IPNewton(linesearch, μ0, show_linesearch)

mutable struct IPNewtonState{T,Tx} <: AbstractBarrierState
    x::Tx
    f_x::T
    x_previous::Tx
    g::Tx
    f_x_previous::T
    H::Matrix{T}          # Hessian of the Lagrangian?
    HP # TODO: remove HP? It's not used
    Hd::Vector{Int8}  # TODO: remove Hd? It's not used
    s::Tx  # step for x
    # Barrier penalty fields
    μ::T                  # coefficient of the barrier penalty
    μnext::T              # μ for the next iteration
    L::T                  # value of the Lagrangian (objective + barrier + equality)
    L_previous::T
    bstate::BarrierStateVars{T}   # value of slack and λ variables (current "position")
    bgrad::BarrierStateVars{T}    # gradient of slack and λ variables at current "position"
    bstep::BarrierStateVars{T}    # search direction for slack and λ
    constr_c::Vector{T}   # value of the user-supplied constraints at x
    constr_J::Matrix{T}   # value of the user-supplied Jacobian at x
    ev::T                 # equality violation, ∑_i λ_Ei (c*_i - c_i)
    Optim.@add_linesearch_fields() # x_ls and alpha
    b_ls::BarrierLineSearchGrad{T}
    gtilde::Tx
    Htilde               # Positive Cholesky factorization of H from PositiveFactorizations.jl
end

# TODO: Do we need this convert thing? (It seems to be used with `show(IPNewtonState)`)
function Base.convert(::Type{IPNewtonState{T,Tx}}, state::IPNewtonState{S, Sx}) where {T,Tx,S,Sx}
    IPNewtonState(convert(Tx, state.x),
                  T(state.f_x),
                  convert(Tx, state.x_previous),
                  convert(Tx, state.g),
                  T(state.f_x_previous),
                  convert(Matrix{T}, state.H),
                  state.HP,
                  state.Hd,
                  convert(Tx, state.s),
                  T(state.μ),
                  T(state.μnext),
                  T(state.L),
                  T(state.L_previous),
                  convert(BarrierStateVars{T}, state.bstate),
                  convert(BarrierStateVars{T}, state.bgrad),
                  convert(BarrierStateVars{T}, state.bstep),
                  convert(Vector{T}, state.constr_c),
                  convert(Matrix{T}, state.constr_J),
                  T(state.ev),
                  convert(Tx, state.x_ls),
                  T(state.alpha),
                  convert(BarrierLineSearchGrad{T}, state.b_ls),
                  convert(Tx, state.gtilde),
                  state.Htilde,
                  )
end

function initial_state(method::IPNewton, options, d::TwiceDifferentiable, constraints::TwiceDifferentiableConstraints, initial_x::Array{T}) where T
    # Check feasibility of the initial state
    mc = nconstraints(constraints)
    constr_c = Array{T}(undef, mc)
    # TODO: When we change to `value!` from NLSolversBase instead of c!
    # we can also update `initial_convergence` for ConstrainedOptimizer in interior.jl
    constraints.c!(constr_c, initial_x)
    if !isinterior(constraints, initial_x, constr_c)
        @warn("Initial guess is not an interior point")
        Base.show_backtrace(stderr, backtrace())
        println(stderr)
    end
    # Allocate fields for the objective function
    n = length(initial_x)
    g = Vector{T}(undef, n)
    s = Vector{T}(undef, n)
    f_x_previous = NaN
    f_x, g_x = value_gradient!(d, initial_x)
    g .= g_x # needs to be a separate copy of g_x
    H = Matrix{T}(undef, n, n)
    Hd = Vector{Int8}(undef, n)
    hessian!(d, initial_x)
    copyto!(H, hessian(d))

    # More constraints
    constr_J = Array{T}(undef, mc, n)
    gtilde = copy(g)
    constraints.jacobian!(constr_J, initial_x)
    μ = T(1)
    bstate = BarrierStateVars(constraints.bounds, initial_x, constr_c)
    bgrad = copy(bstate)
    bstep = copy(bstate)
    # b_ls = BarrierLineSearch(similar(constr_c), similar(bstate))
    b_ls = BarrierLineSearchGrad(copy(constr_c), copy(constr_J), copy(bstate), copy(bstate))

    state = IPNewtonState(
        copy(initial_x), # Maintain current state in state.x
        f_x, # Store current f in state.f_x
        copy(initial_x), # Maintain previous state in state.x_previous
        g, # Store current gradient in state.g (TODO: includes Lagrangian calculation?)
        T(NaN), # Store previous f in state.f_x_previous
        H,
        0,    # will be replaced
        Hd,
        similar(initial_x), # Maintain current x-search direction in state.s
        μ,
        μ,
        T(NaN),
        T(NaN),
        bstate,
        bgrad,
        bstep,
        constr_c,
        constr_J,
        T(NaN),
        Optim.@initial_linesearch()..., # Maintain a cache for line search results in state.lsr
        b_ls,
        gtilde,
        0)

    Hinfo = (state.H, hessianI(initial_x, constraints, 1 ./ bstate.slack_c, 1))
    initialize_μ_λ!(state, constraints.bounds, Hinfo, method.μ0)
    update_fg!(d, constraints, state, method)
    update_h!(d, constraints, state, method)

    state
end

function update_fg!(d, constraints::TwiceDifferentiableConstraints, state, method::IPNewton)
    state.f_x, state.L, state.ev = lagrangian_fg!(state.g, state.bgrad, d, constraints.bounds, state.x, state.constr_c, state.constr_J, state.bstate, state.μ)
    update_gtilde!(d, constraints, state, method)
end

function update_gtilde!(d, constraints::TwiceDifferentiableConstraints, state, method::IPNewton)
    # Calculate the modified x-gradient for the block-eliminated problem
    # gtilde is the gradient for the affine-scaling problem, i.e.,
    # with μ=0, used in the adaptive setting of μ. Once we calculate μ we'll correct it
    gtilde, bstate, bgrad = state.gtilde, state.bstate, state.bgrad
    bounds = constraints.bounds
    copyto!(gtilde, state.g)
    JIc = view(state.constr_J, bounds.ineqc, :)
    if !isempty(JIc)
        Hssc = Diagonal(bstate.λc./bstate.slack_c)
        # TODO: Can we use broadcasting / dot-notation here and eliminate gc?
        gc = JIc'*(Diagonal(bounds.σc) * (bstate.λc - Hssc*bgrad.λc))  # NOT bgrad.slack_c
        gtilde .+= gc
    end
    for (i,j) in enumerate(bounds.ineqx)
        gxi = bounds.σx[i]*(bstate.λx[i] -  bgrad.λx[i]*bstate.λx[i]/bstate.slack_x[i])
        gtilde[j] += gxi
    end
    state
end

function update_h!(d, constraints::TwiceDifferentiableConstraints, state, method::IPNewton)
    x, μ, Hxx, J = state.x, state.μ, state.H, state.constr_J
    bstate, bgrad, bounds = state.bstate, state.bgrad, constraints.bounds
    m, n = size(J, 1), size(J, 2)

    hessian!(d, state.x)  # objective's Hessian
    copyto!(Hxx, hessian(d))  # objective's Hessian
    # accumulate the constraint second derivatives
    λ = userλ(bstate.λc, constraints)
    λ[bounds.eqc] = -bstate.λcE  # the negative sign is from the Hessian
    # Important! We are assuming that constraints.h! adds the hessian of the
    # non-objective Lagrangian terms to the existing objective Hessian Hxx.
    # This follows the approach by the CUTEst interface
    constraints.h!(Hxx, x, λ)

    # Add the Jacobian terms (JI'*Hss*JI)
    JIc = view(J, bounds.ineqc, :)
    Hssc = Diagonal(bstate.λc./bstate.slack_c)
    HJ = JIc'*Hssc*JIc
    for j = 1:n, i = 1:n
        Hxx[i,j] += HJ[i,j]
    end
    # Add the variable inequalities portions of J'*Hssx*J
    for (i,j) in enumerate(bounds.ineqx)
        Hxx[j,j] += bstate.λx[i]/bstate.slack_x[i]
    end
    state.Htilde = cholesky(Positive, Hxx, Val{true})

    state
end

function update_state!(d, constraints::TwiceDifferentiableConstraints, state::IPNewtonState{T}, method::IPNewton, options) where T
    state.f_x_previous, state.L_previous = state.f_x, state.L
    bstate, bstep, bounds = state.bstate, state.bstep, constraints.bounds
    qp = solve_step!(state, constraints, options, method.show_linesearch)
    # If a step α=1 will not change any of the parameters, we can quit now.
    # This prevents a futile linesearch.
    if is_smaller_eps(state.x, state.s) &&
        is_smaller_eps(bstate.slack_x, bstep.slack_x) &&
        is_smaller_eps(bstate.slack_c, bstep.slack_c) &&
        is_smaller_eps(bstate.λx, bstep.λx) &&
        is_smaller_eps(bstate.λc, bstep.λc)
        return false
    end

    # Estimate αmax, the upper bound on distance of movement along the search line
    αmax = convert(eltype(bstate), Inf)
    αmax = estimate_maxstep(αmax, bstate.slack_x, bstep.slack_x)
    αmax = estimate_maxstep(αmax, bstate.slack_c, bstep.slack_c)
    αmax = estimate_maxstep(αmax, bstate.λx, bstep.λx)
    αmax = estimate_maxstep(αmax, bstate.λc, bstep.λc)

    # Determine the actual distance of movement along the search line
    ϕ = linesearch_anon(d, constraints, state, method)
    # TODO: This only works for method.linesearch = backtracking_constrained_grad
    # TODO: How are we meant to implement backtracking_constrained?.
    #       It requires both an alpha and an alphaI (αmax and αImax) ...
    state.alpha =
        method.linesearch!(ϕ, T(1), αmax, qp; show_linesearch=method.show_linesearch)

    # Maintain a record of previous position
    copyto!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    ls_update!(state.x, state.x, state.s, state.alpha)
    ls_update!(bstate, bstate, bstep, state.alpha)

    # Ensure that the primal-dual approach does not deviate too much from primal
    # (See Waechter & Biegler 2006, eq. 16)
    μ = state.μ
    for i = 1:length(bstate.slack_x)
        p = μ / bstate.slack_x[i]
        bstate.λx[i] = max(min(bstate.λx[i], 1e10*p), 1e-10*p)
    end
    for i = 1:length(bstate.slack_c)
        p = μ / bstate.slack_c[i]
        bstate.λc[i] = max(min(bstate.λc[i], 1e10*p), 1e-10*p)
    end
    state.μ = state.μnext

    # Evaluate the constraints at the new position
    constraints.c!(state.constr_c, state.x)
    constraints.jacobian!(state.constr_J, state.x)
    state.ev == equality_violation(constraints, state)

    false
end

function solve_step!(state::IPNewtonState, constraints, options, show_linesearch::Bool = false)
    x, s, μ, bounds = state.x, state.s, state.μ, constraints.bounds
    bstate, bstep, bgrad = state.bstate, state.bstep, state.bgrad
    J, Htilde = state.constr_J, state.Htilde
    # Solve the Newton step
    JE = jacobianE(state, bounds)
    gE = [bgrad.λxE;
          bgrad.λcE]
    M = JE*(Htilde \ JE')
    MF = cholesky(Positive, M, Val{true})
    # These are a solution to the affine-scaling problem (with μ=0)
    ΔλE0 = MF \ (gE + JE * (Htilde \ state.gtilde))
    Δx0 = Htilde \ (JE'*ΔλE0 - state.gtilde)
    # Check that the solution to the linear equations represents an improvement
    Hpstepx, HstepλE = Matrix(Htilde)*Δx0 - JE'*ΔλE0, -JE*Δx0  # TODO: don't use full here
    # TODO: How to handle show_linesearch?
    # This was originally in options.show_linesearch, but I removed it as none of the other Optim algorithms have it there.
    # We should move show_linesearch back to options when we refactor
    # LineSearches to work on the function ϕ(α)
    if show_linesearch
        println("|gx| = $(norm(state.gtilde)), |Hstepx + gx| = $(norm(Hpstepx+state.gtilde))")
        println("|gE| = $(norm(gE)), |HstepλE + gE| = $(norm(HstepλE+gE))")
    end
    if norm(gE) + norm(state.gtilde) < max(norm(HstepλE + gE),
                                           norm(Hpstepx  + state.gtilde))
        # Precision problems gave us a worse solution than the one we started with, abort
        fill!(s, 0)
        fill!(bstep, 0)
        return state
    end
    # Set μ (see the predictor strategy in Nodecal & Wright, 2nd ed., section 19.3)
    solve_slack!(bstep, Δx0, bounds, bstate, bgrad, J, zero(state.μ)) # store temporarily in bstep
    αs = convert(eltype(bstate), 1.0)
    αs = estimate_maxstep(αs, bstate.slack_x, bstep.slack_x)
    αs = estimate_maxstep(αs, bstate.slack_c, bstep.slack_c)
    αλ = convert(eltype(bstate), 1.0)
    αλ = estimate_maxstep(αλ, bstate.λx, bstep.λx)
    αλ = estimate_maxstep(αλ, bstate.λc, bstep.λc)
    m = max(1, length(bstate.slack_x) + length(bstate.slack_c))
    μaff = (dot(bstate.slack_x + αs*bstep.slack_x, bstate.λx + αλ*bstep.λx) +
            dot(bstate.slack_c + αs*bstep.slack_c, bstate.λc + αλ*bstep.λc))/m
    μmean = (dot(bstate.slack_x, bstate.λx) + dot(bstate.slack_c, bstate.λc))/m
    # When there's only one constraint, μaff can be exactly zero. So limit the decrease.
    state.μnext = NaNMath.max((μaff/μmean)^3 * μmean, μmean/10)
    μ = state.μ
    # Solve for the *real* step (including μ)
    μsinv = μ * [bounds.σx./bstate.slack_x; bounds.σc./bstate.slack_c]
    gtildeμ = state.gtilde  - jacobianI(state, bounds)' * μsinv
    ΔλE = MF \ (gE + JE * (Htilde \ gtildeμ))
    Δx = Htilde \ (JE'*ΔλE - gtildeμ)
    copyto!(s, Δx)
    k = unpack_vec!(bstep.λxE, ΔλE, 0)
    k = unpack_vec!(bstep.λcE, ΔλE, k)
    k == length(ΔλE) || error("exhausted targets before ΔλE")
    solve_slack!(bstep, Δx, bounds, bstate, bgrad, J, μ)
    # Solve for the quadratic parameters (use the real H, not the posdef H)
    Hstepx, HstepλE  = state.H*Δx - JE'*ΔλE, -JE*Δx
    qp = state.L, slopealpha(state.s, state.g, bstep, bgrad), dot(Δx, Hstepx) + dot(ΔλE, HstepλE)
    qp
end

function solve_slack!(bstep, s, bounds, bstate, bgrad, J, μ)
    # Solve for the slack variable and λI updates
    for (i, j) in enumerate(bounds.ineqx)
        bstep.slack_x[i] = -bgrad.λx[i] + bounds.σx[i]*s[j]
        # bstep.λx[i] = -bgrad.slack_x[i] - μ*bstep.slack_x[i]/bstate.slack_x[i]^2
        # bstep.λx[i] = -bgrad.slack_x[i] - bstate.λx[i]*bstep.slack_x[i]/bstate.slack_x[i]
        bstep.λx[i] = -(-μ/bstate.slack_x[i] + bstate.λx[i]) - bstate.λx[i]*bstep.slack_x[i]/bstate.slack_x[i]
    end
    JIc = view(J, bounds.ineqc, :)
    SigmaJIΔx = Diagonal(bounds.σc)*(JIc*s)
    for i = 1:length(bstep.λc)
        bstep.slack_c[i] = -bgrad.λc[i] + SigmaJIΔx[i]
        # bstep.λc[i] = -bgrad.slack_c[i] - μ*bstep.slack_c[i]/bstate.slack_c[i]^2
        # bstep.λc[i] = -bgrad.slack_c[i] - bstate.λc[i]*bstep.slack_c[i]/bstate.slack_c[i]
        bstep.λc[i] = -(-μ/bstate.slack_c[i] + bstate.λc[i]) - bstate.λc[i]*bstep.slack_c[i]/bstate.slack_c[i]
    end
    bstep
end

function is_smaller_eps(ref, step)
    ise = true
    for (r, s) in zip(ref, step)
        ise &= (s == 0) | (abs(s) < eps(r))
    end
    ise
end

function default_options(method::ConstrainedOptimizer)
    Dict(:allow_f_increases => true, :successive_f_tol => 2)
end


# Utility functions that assist in testing: they return the "full
# Hessian" and "full gradient" for the equation with the slack and λI
# eliminated.
# TODO: should we put these elsewhere?
function Hf(bounds::ConstraintBounds, state)
    JE = jacobianE(state, bounds)
    Hf = [Matrix(state.Htilde) -JE';
          -JE zeros(eltype(JE), size(JE, 1), size(JE, 1))]
end
Hf(constraints, state) = Hf(constraints.bounds, state)
function gf(bounds::ConstraintBounds, state)
    bstate, μ = state.bstate, state.μ
    μsinv = μ * [bounds.σx./bstate.slack_x; bounds.σc./bstate.slack_c]
    gtildeμ = state.gtilde  - jacobianI(state, bounds)' * μsinv
    [gtildeμ; state.bgrad.λxE; state.bgrad.λcE]
end
gf(constraints, state) = gf(constraints.bounds, state)
