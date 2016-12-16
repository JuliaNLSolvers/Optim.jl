immutable IPNewton{F} <: IPOptimizer{F}
    linesearch!::F
end

IPNewton(; linesearch!::Function = backtrack_constrained_grad) =
  IPNewton(linesearch!)

type IPNewtonState{T,N} <: AbstractBarrierState
    @add_generic_fields()
    x_previous::Array{T,N}
    g::Array{T,N}
    f_x_previous::T
    H::Matrix{T}
    HP
    Hd::Vector{Int8}
    s::Array{T,N}  # step for x
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
    @add_linesearch_fields()
    b_ls::BarrierLineSearchGrad{T}
    gtilde::Vector{T}
    Htilde
end

function Base.convert{T,S,N}(::Type{IPNewtonState{T,N}}, state::IPNewtonState{S,N})
    IPNewtonState(state.method_string,
                  state.n,
                  convert(Array{T}, state.x),
                  T(state.f_x),
                  state.f_calls,
                  state.g_calls,
                  state.h_calls,
                  convert(Array{T}, state.x_previous),
                  convert(Array{T}, state.g),
                  T(state.f_x_previous),
                  convert(Array{T}, state.H),
                  state.HP,
                  state.Hd,
                  convert(Array{T}, state.s),
                  T(state.μ),
                  T(state.μnext),
                  T(state.L),
                  T(state.L_previous),
                  convert(BarrierStateVars{T}, state.bstate),
                  convert(BarrierStateVars{T}, state.bgrad),
                  convert(BarrierStateVars{T}, state.bstep),
                  convert(Array{T}, state.constr_c),
                  convert(Array{T}, state.constr_J),
                  T(state.ev),
                  convert(Array{T}, state.x_ls),
                  convert(Array{T}, state.g_ls),
                  T(state.alpha),
                  state.mayterminate,
                  state.lsr,
                  convert(BarrierLineSearchGrad{T}, state.b_ls),
                  convert(Array{T}, state.gtilde),
                  state.Htilde,
                  )
end

function initial_state{T}(method::IPNewton, options, d::TwiceDifferentiableFunction, constraints::TwiceDifferentiableConstraintsFunction, initial_x::Array{T})
    # Check feasibility of the initial state
    mc = nconstraints(constraints)
    constr_c = Array{T}(mc)
    constraints.c!(initial_x, constr_c)
    if !isinterior(constraints, initial_x, constr_c)
        warn("initial guess is not an interior point")
        Base.show_backtrace(STDERR, backtrace())
        println(STDERR)
    end
    # Allocate fields for the objective function
    n = length(initial_x)
    g = Array(T, n)
    s = Array(T, n)
    x_ls, g_ls = Array(T, n), Array(T, n)
    f_x_previous, f_x = NaN, d.fg!(initial_x, g)
    f_calls, g_calls = 1, 1
    H = Array(T, n, n)
    Hd = Array{Int8}(n)
    d.h!(initial_x, H)
    h_calls = 1

    # More constraints
    constr_J = Array{T}(mc, n)
    constr_gtemp = Array{T}(n)
    gtilde = similar(g)
    constraints.jacobian!(initial_x, constr_J)
    μ = T(1)
    bstate = BarrierStateVars(constraints.bounds, initial_x, constr_c)
    bgrad = similar(bstate)
    bstep = similar(bstate)
    # b_ls = BarrierLineSearch(similar(constr_c), similar(bstate))
    b_ls = BarrierLineSearchGrad(similar(constr_c), similar(constr_J), similar(bstate), similar(bstate))

    state = IPNewtonState("Interior-point Newton's Method",
        length(initial_x),
        copy(initial_x), # Maintain current state in state.x
        f_x, # Store current f in state.f_x
        f_calls, # Track f calls in state.f_calls
        g_calls, # Track g calls in state.g_calls
        h_calls,
        copy(initial_x), # Maintain current state in state.x_previous
        g, # Store current gradient in state.g
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
        @initial_linesearch()..., # Maintain a cache for line search results in state.lsr
        b_ls,
        gtilde,
        0)

    d.h!(initial_x, state.H)
    Hinfo = (state.H, hessianI(initial_x, constraints, 1./bstate.slack_c, 1))
    initialize_μ_λ!(state, constraints.bounds, Hinfo, options.μ0)
    update_fg!(d, constraints, state, method)
    update_h!(d, constraints, state, method)
end

function update_fg!(d, constraints::TwiceDifferentiableConstraintsFunction, state, method::IPNewton)
    state.f_x, state.L, state.ev = lagrangian_fg!(state.g, state.bgrad, d, constraints.bounds, state.x, state.constr_c, state.constr_J, state.bstate, state.μ)
    state.f_calls += 1
    state.g_calls += 1
    update_gtilde!(d, constraints, state, method)
end

function update_g!(d, constraints::TwiceDifferentiableConstraintsFunction, state, method::IPNewton)
    lagrangian_g!(state.g, state.bgrad, d, constraints.bounds, state.x, state.constr_c, state.constr_J, state.bstate, state.μ)
    state.g_calls += 1
    update_gtilde!(d, constraints, state, method)
end

function update_gtilde!(d, constraints::TwiceDifferentiableConstraintsFunction, state, method::IPNewton)
    # Calculate the modified x-gradient for the block-eliminated problem
    # gtilde is the gradient for the affine-scaling problem, i.e.,
    # with μ=0, used in the adaptive setting of μ. Once we calculate μ we'll correct it
    gtilde, bstate, bgrad = state.gtilde, state.bstate, state.bgrad
    bounds = constraints.bounds
    copy!(gtilde, state.g)
    JIc = view5(state.constr_J, bounds.ineqc, :)
    if !isempty(JIc)
        Hssc = Diagonal(bstate.λc./bstate.slack_c)
        gc = JIc'*(Diagonal(bounds.σc) * (bstate.λc - Hssc*bgrad.λc))  # NOT bgrad.slack_c
        for i = 1:length(gtilde)
            gtilde[i] += gc[i]
        end
    end
    for (i,j) in enumerate(bounds.ineqx)
        gxi = bounds.σx[i]*(bstate.λx[i] -  bgrad.λx[i]*bstate.λx[i]/bstate.slack_x[i])
        gtilde[j] += gxi
    end
    state
end

function update_h!(d, constraints::TwiceDifferentiableConstraintsFunction, state, method::IPNewton)
    x, μ, Hxx, J = state.x, state.μ, state.H, state.constr_J
    bstate, bgrad, bounds = state.bstate, state.bgrad, constraints.bounds
    m, n = size(J, 1), size(J, 2)

    d.h!(state.x, Hxx)  # objective's Hessian
    # accumulate the constraint second derivatives
    λ = userλ(bstate.λc, constraints)
    λ[bounds.eqc] = -bstate.λcE  # the negative sign is from the Hessian
    constraints.h!(x, λ, Hxx)
    # Add the Jacobian terms (JI'*Hss*JI)
    JIc = view5(J, bounds.ineqc, :)
    Hssc = Diagonal(bstate.λc./bstate.slack_c)
    HJ = JIc'*Hssc*JIc
    for j = 1:n, i = 1:n
        Hxx[i,j] += HJ[i,j]
    end
    # Add the variable inequalities portions of J'*Hssx*J
    for (i,j) in enumerate(bounds.ineqx)
        Hxx[j,j] += bstate.λx[i]/bstate.slack_x[i]
    end
    state.Htilde = cholfact(Positive, state.H, Val{true})

    state
end

function update_state!{T}(d, constraints::TwiceDifferentiableConstraintsFunction, state::IPNewtonState{T}, method::IPNewton, options)
    state.f_x_previous, state.L_previous = state.f_x, state.L
    bstate, bstep, bounds = state.bstate, state.bstep, constraints.bounds
    qp = solve_step!(state, constraints, options)
    # If a step α=1 will not change any of the parameters, we can quit now.
    # This prevents a futile linesearch.
    if is_smaller_eps(state.x, state.s) &&
        is_smaller_eps(bstate.slack_x, bstep.slack_x) &&
        is_smaller_eps(bstate.slack_c, bstep.slack_c) &&
        is_smaller_eps(bstate.λx, bstep.λx) &&
        is_smaller_eps(bstate.λc, bstep.λc)
        return false
    end
    # qp = quadratic_parameters(bounds, state)

    # Estimate αmax, the upper bound on distance of movement along the search line
    αmax = convert(eltype(bstate), Inf)
    αmax = estimate_maxstep(αmax, bstate.slack_x, bstep.slack_x)
    αmax = estimate_maxstep(αmax, bstate.slack_c, bstep.slack_c)
    αmax = estimate_maxstep(αmax, bstate.λx, bstep.λx)
    αmax = estimate_maxstep(αmax, bstate.λc, bstep.λc)

    # Determine the actual distance of movement along the search line
    ϕ = linesearch_anon(d, constraints, state, method)
    state.alpha, f_update, g_update =
        method.linesearch!(ϕ, T(1), αmax, qp; show_linesearch=options.show_linesearch)
    state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    ls_update!(state.x, state.x, state.s, state.alpha)
    ls_update!(bstate, bstate, bstep, state.alpha)

    # Ensure that the primal-dual approach does not deviate too much from primal
    # (See Waechter & Biegler 2006, eq. 16)
    μ = state.μ
    for i = 1:length(bstate.slack_x)
        p = μ/bstate.slack_x[i]
        bstate.λx[i] = max(min(bstate.λx[i], 10^10*p), p/10^10)
    end
    for i = 1:length(bstate.slack_c)
        p = μ/bstate.slack_c[i]
        bstate.λc[i] = max(min(bstate.λc[i], 10^10*p), p/10^10)
    end
    state.μ = state.μnext

    # Evaluate the constraints at the new position
    constraints.c!(state.x, state.constr_c)
    constraints.jacobian!(state.x, state.constr_J)
    state.ev == equality_violation(constraints, state)

    false
end

function solve_step!(state::IPNewtonState, constraints, options)
    x, s, μ, bounds = state.x, state.s, state.μ, constraints.bounds
    bstate, bstep, bgrad = state.bstate, state.bstep, state.bgrad
    J, Htilde = state.constr_J, state.Htilde
    # Solve the Newton step
    JE = jacobianE(state, bounds)
    gE = [bgrad.λxE;
          bgrad.λcE]
    M = JE*(Htilde \ JE')
    MF = cholfact(Positive, M, Val{true})
    # These are a solution to the affine-scaling problem (with μ=0)
    ΔλE0 = MF \ (gE + JE * (Htilde \ state.gtilde))
    Δx0 = Htilde \ (JE'*ΔλE0 - state.gtilde)
    # Check that the solution to the linear equations represents an improvement
    Hpstepx, HstepλE = full(Htilde)*Δx0 - JE'*ΔλE0, -JE*Δx0  # TODO: don't use full here
    if options.show_linesearch
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
    state.μnext = max((μaff/μmean)^3 * μmean, μmean/10)
    μ = state.μ
    # Solve for the *real* step (including μ)
    μsinv = μ * [bounds.σx./bstate.slack_x; bounds.σc./bstate.slack_c]
    gtildeμ = state.gtilde  - jacobianI(state, bounds)' * μsinv
    ΔλE = MF \ (gE + JE * (Htilde \ gtildeμ))
    Δx = Htilde \ (JE'*ΔλE - gtildeμ)
    copy!(s, Δx)
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
    JIc = view5(J, bounds.ineqc, :)
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

"""
    quadratic_parameters(bounds, state) -> val, slopeα, Hα

OUTDATED! Return the parameters for the quadratic fit of the behavior of the
lagrangian for positions parametrized as a function of the 4-vector
`α = (αx, αs, αI, αE)`, where the step is

    (αx * Δx, αs * Δs, αI * ΔλI, αE * ΔλE)

and `Δx`, `Δs`, `ΔλI`, and `ΔλE` are the current search directions in
the parameters. As a function of `α`, the local model is expressed as

    val + dot(α, slopeα) + (α'*Hα*α)/2
"""
function quadratic_parameters(bounds::ConstraintBounds, state::IPNewtonState)
    bstate, bstep, bgrad = state.bstate, state.bstep, state.bgrad
    slopeα = slopealpha(state.s, state.g, bstep, bgrad)

    jic = view5(state.constr_J, bounds.ineqc, :)*state.s
    HsscP = Diagonal(state.μ./bstate.slack_c.^2)  # for linesearch we need primal
    jix = view(state.s, bounds.ineqx)
    HssxP = Diagonal(state.μ./bstate.slack_x.^2)
    ji = dot(bstep.λc, Diagonal(bounds.σc)*jic) + dot(bstep.λx, Diagonal(bounds.σx)*jix)
    je = dot(bstep.λcE, view5(state.constr_J, bounds.eqc, :)*state.s) +
         dot(bstep.λxE, view(state.s, bounds.eqx))
    hss = dot(bstep.slack_c, HsscP*bstep.slack_c) + dot(bstep.slack_x, HssxP*bstep.slack_x)
    si = dot(bstep.slack_c, bstep.λc) + dot(bstep.slack_x, bstep.λx)
    hxx = dot(state.s, full(state.HP)*state.s)  # TODO: don't require full here
    Hα = [hxx    0    -ji   -je;
          0      hss  si    0;
          -ji    si   0     0;
          -je    0    0     0]
    state.L, slopeα, Hα
end

# Utility functions that assist in testing: they return the "full
# Hessian" and "full gradient" for the equation with the slack and λI
# eliminated.
function Hf(bounds::ConstraintBounds, state)
    JE = jacobianE(state, bounds)
    Hf = [full(state.Htilde) -JE';
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
