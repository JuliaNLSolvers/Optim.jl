immutable IPNewton{F} <: IPOptimizer{F}
    linesearch!::F
end

IPNewton(; linesearch!::Function = backtrack_constrained) =
  IPNewton(linesearch!)

type IPNewtonState{T,N} <: AbstractBarrierState
    @add_generic_fields()
    x_previous::Array{T,N}
    g::Array{T,N}
    f_x_previous::T
    H::Matrix{T}
    Hd::Vector{Int8}
    s::Array{T,N}  # step for x
    # Barrier penalty fields
    μ::T                  # coefficient of the barrier penalty
    L::T                  # value of the Lagrangian (objective + barrier + equality)
    L_previous::T
    bstate::BarrierStateVars{T}   # value of slack and λ variables (current "position")
    bgrad::BarrierStateVars{T}    # gradient of slack and λ variables at current "position"
    bstep::BarrierStateVars{T}    # search direction for slack and λ
    constr_c::Vector{T}   # value of the user-supplied constraints at x
    constr_J::Matrix{T}   # value of the user-supplied Jacobian at x
    ev::T                 # equality violation, ∑_i λ_Ei (c*_i - c_i)
    @add_linesearch_fields()
    b_ls::BarrierLineSearch{T}
    gtilde::Vector{T}
end

function initial_state{T}(method::IPNewton, options, d::TwiceDifferentiableFunction, constraints::TwiceDifferentiableConstraintsFunction, initial_x::Array{T})
    # Check feasibility of the initial state
    mc = nconstraints(constraints)
    constr_c = Array{T}(mc)
    constraints.c!(initial_x, constr_c)
    isinterior(constraints, initial_x, constr_c) || (warn("initial guess is not an interior point"); Base.show_backtrace(STDOUT, backtrace()))

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
    b_ls = BarrierLineSearch(similar(constr_c), similar(bstate))

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
        Hd,
        similar(initial_x), # Maintain current x-search direction in state.s
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
        gtilde)

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
    gtilde, bstate, bgrad = state.gtilde, state.bstate, state.bgrad
    bounds = constraints.bounds
    copy!(gtilde, state.g)
    JIc = view5(state.constr_J, bounds.ineqc, :)
    if !isempty(JIc)
        Hssc = Diagonal(bstate.λc./bstate.slack_c)
        gc = JIc'*(Diagonal(bounds.σc) * (bgrad.slack_c - Hssc*bgrad.λc))
        for i = 1:length(gtilde)
            gtilde[i] += gc[i]
        end
    end
    for (i,j) in enumerate(bounds.ineqx)
        gxi = bounds.σx[i]*(bgrad.slack_x[i] -  bgrad.λx[i]*bstate.λx[i]/bstate.slack_x[i])
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
    # Add the Jacobian terms (J'*Hss*J)
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

    state
end

function update_state!{T}(d, constraints::TwiceDifferentiableConstraintsFunction, state::IPNewtonState{T}, method::IPNewton)
    state.f_x_previous, state.L_previous = state.f_x, state.L
    bstate, bstep, bounds = state.bstate, state.bstep, constraints.bounds
    state = solve_step!(state, constraints)
    # If a step α=1 will not change any of the parameters, we can quit now.
    # This prevents a futile linesearch.
    if is_smaller_eps(state.x, state.s) &&
        is_smaller_eps(bstate.slack_x, bstep.slack_x) &&
        is_smaller_eps(bstate.slack_c, bstep.slack_c) &&
        is_smaller_eps(bstate.λx, bstep.λx) &&
        is_smaller_eps(bstate.λc, bstep.λc)
        return false
    end
    qp = quadratic_parameters(bounds, state)

    # Estimate αmax, the upper bound on distance of movement along the search line
    αmax = αImax = convert(eltype(bstate), Inf)
    αmax = estimate_maxstep(αmax, bstate.slack_x, bstep.slack_x)
    αmax = estimate_maxstep(αmax, bstate.slack_c, bstep.slack_c)
    αImax = estimate_maxstep(αImax, bstate.λx, bstep.λx)
    αImax = estimate_maxstep(αImax, bstate.λc, bstep.λc)

    # Determine the actual distance of movement along the search line
    ϕ = (α,αI)->lagrangian_linefunc!(α, αI, d, constraints, state, method)
    state.alpha, αI, f_update, g_update =
        method.linesearch!(ϕ, T(1), αmax, αImax, qp)
    state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    ls_update!(state.x, state.x, state.s, state.alpha)
    ls_update!(bstate, bstate, bstep, state.alpha, αI)

    # Evaluate the constraints at the new position
    constraints.c!(state.x, state.constr_c)
    constraints.jacobian!(state.x, state.constr_J)
    @assert state.ev == equality_violation(constraints, state)

    false
end

function solve_step!(state::IPNewtonState, constraints)
    x, s, μ, bounds = state.x, state.s, state.μ, constraints.bounds
    bstate, bstep, bgrad = state.bstate, state.bstep, state.bgrad
    # Solve the Newton step
    Hxx = state.H
    JE = jacobianE(state, bounds)
    # Q, R, p = qr(JE', Val{true})
    gE = [bgrad.λxE;
          bgrad.λcE]
    HxxF = cholfact(Positive, Hxx, Val{true})
    M = JE*(HxxF \ JE')
    MF = cholfact(Positive, M, Val{true})
    ΔλE = MF \ (gE + JE * (HxxF \ state.gtilde))
    Δx = HxxF \ (JE'*ΔλE - state.gtilde)
    copy!(s, Δx)
    k = unpack_vec!(bstep.λxE, ΔλE, 0)
    k = unpack_vec!(bstep.λcE, ΔλE, k)
    k == length(ΔλE) || error("exhausted targets before ΔλE")
    # Solve for the slack variable and λI updates
    for (i, j) in enumerate(bounds.ineqx)
        bstep.slack_x[i] = -bgrad.λx[i] + bounds.σx[i]*s[j]
        # bstep.λx[i] = -bgrad.slack_x[i] - μ*bstep.slack_x[i]/bstate.slack_x[i]^2
        bstep.λx[i] = -bgrad.slack_x[i] - bstate.λx[i]*bstep.slack_x[i]/bstate.slack_x[i]
    end
    JIc = view5(state.constr_J, bounds.ineqc, :)
    SigmaJIΔx = Diagonal(bounds.σc)*(JIc*state.s)
    for i = 1:length(bstep.λc)
        bstep.slack_c[i] = -bgrad.λc[i] + SigmaJIΔx[i]
        # bstep.λc[i] = -bgrad.slack_c[i] - μ*bstep.slack_c[i]/bstate.slack_c[i]^2
        bstep.λc[i] = -bgrad.slack_c[i] - bstate.λc[i]*bstep.slack_c[i]/bstate.slack_c[i]
    end
    state
end

function is_smaller_eps(ref, step)
    ise = true
    for (r, s) in zip(ref, step)
        ise &= (s == 0) | (abs(s) < eps(r))
    end
    ise
end

function quadratic_parameters(bounds::ConstraintBounds, state::IPNewtonState)
    slope = dot(state.s, state.gtilde) +
        dot(state.bstep.λxE, state.bgrad.λxE) +
        dot(state.bstep.λcE, state.bgrad.λcE)
    # For the curvature, use the original hessian (before forcing
    # positive-definiteness)
    q = dot(state.s, state.H*state.s)
    JE = view5(state.constr_J, bounds.eqc, :)
    q -= 2*dot(state.s[bounds.eqx], state.bstep.λxE) + 2*dot(state.s, JE'*state.bstep.λcE)
    state.L, slope, q
end

# Utility functions that assist in testing: they return the "full
# Hessian" and "full gradient" for the equation with the slack and λI
# eliminated.
function Hf(bounds::ConstraintBounds, state)
    JE = jacobianE(state, bounds)
    HxxF = cholfact(Positive, state.H)
    Hf = [full(HxxF) -JE';
          -JE zeros(eltype(JE), size(JE, 1), size(JE, 1))]
end
Hf(constraints, state) = Hf(constraints.bounds, state)
gf(state) = [state.gtilde; state.bgrad.λxE; state.bgrad.λcE]
