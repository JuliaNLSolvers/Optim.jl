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
    bstate::BarrierStateVars{T}   # value of slack and λ variables (current "position")
    bgrad::BarrierStateVars{T}    # gradient of slack and λ variables at current "position"
    bstep::BarrierStateVars{T}    # search direction for slack and λ
    constr_c::Vector{T}   # value of the user-supplied constraints at x
    constr_J::Matrix{T}   # value of the user-supplied Jacobian at x
    @add_linesearch_fields()
    b_ls::BarrierLineSearch{T}
    gf::Vector{T}
    Hf::Matrix{T}
    stepf::Vector{T}
end

function initial_state{T}(method::IPNewton, options, d::TwiceDifferentiableFunction, constraints::TwiceDifferentiableConstraintsFunction, initial_x::Array{T})
    # Check feasibility of the initial state
    mc = nconstraints(constraints)
    constr_c = Array{T}(mc)
    constraints.c!(initial_x, constr_c)
#    isfeasible(constraints, initial_x, constr_c) || error("initial guess must be feasible")

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
    gf = Array{T}(0)    # will be replaced
    Hf = Array{T}(0,0)  #   "
    stepf = Array{T}(0)
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
        T(0),
        bstate,
        bgrad,
        bstep,
        constr_c,
        constr_J,
        @initial_linesearch()..., # Maintain a cache for line search results in state.lsr
        b_ls,
        gf,
        Hf,
        stepf)

    d.h!(initial_x, state.H)
    # state.μ = initialize_μ_λ!(bstate.λxE, bstate.λcE, constraints, initial_x, g, constr_c, constr_J, options.μ0)
    Hinfo = (state.H, hessianI(initial_x, constraints, 1./bstate.slack_c, 1))
    initialize_μ_λ!(state, constraints.bounds, Hinfo, options.μ0)
    update_fg!(d, constraints, state, method)
    update_h!(d, constraints, state, method)
end

function update_fg!(d, constraints::TwiceDifferentiableConstraintsFunction, state, method::IPNewton)
    f_x, L = lagrangian_fg!(state.g, state.bgrad, d, constraints.bounds, state.x, state.constr_c, state.constr_J, state.bstate, state.μ)
    state.f_x, state.L = f_x, L
    state.f_calls += 1
    state.g_calls += 1
    state
end

function update_g!(d, constraints::TwiceDifferentiableConstraintsFunction, state, method::IPNewton)
    lagrangian_g!(state.g, state.bgrad, d, constraints.bounds, state.x, state.constr_c, state.constr_J, state.bstate, state.μ)
    state.g_calls += 1
    state
end

function update_h!(d, constraints::TwiceDifferentiableConstraintsFunction, state, method::IPNewton)
    x = state.x
    μ, Hxx, J = state.μ, state.H, state.constr_J
    bounds = constraints.bounds
    m, n = size(J, 1), size(J, 2)

    d.h!(state.x, Hxx)
    hessianI!(Hxx, state.x, constraints, state.bstate.λc, μ)  # accumulate the inequality second derivatives
    # Add the Jacobian terms (J'*S^{-2}*J)
    JI = view5(J, bounds.ineqc, :)
    Sinv2 = Diagonal(1./state.bstate.slack_c.^2)
    HJ = JI'*Sinv2*JI
    for j = 1:n, i = 1:n
        Hxx[i,j] += μ*HJ[i,j]
    end
    # Add the variable inequalities portions of J'*S^{-2}*J
    # The iz terms are already in Hxx (from hessianI!)
    ineqx, sx = bounds.ineqx, state.bstate.slack_x
    for (i,j) in enumerate(ineqx)
        Hxx[j,j] += μ/sx[i]^2
    end
    # Perform a positive factorization
    Hpc, state.Hd = ldltfact(Positive, Hxx)
    Hp = full(Hpc)
    # Now add the equality constraint hessian terms
    eqc, λcE = bounds.eqc, state.bstate.λcE
    λ = zeros(eltype(x), nconstraints(bounds))
    for i = 1:length(eqc)
        λ[eqc[i]] -= λcE[i]
    end
    constraints.h!(state.x, λ, Hp)
    # Also add these to Hxx so we have the true Hessian (the one
    # without forcing positive-definiteness)
    constraints.h!(state.x, λ, Hxx)
    # Form the total Hessian
    JEx = zeros(eltype(bounds), length(bounds.eqx), length(state.x))
    for (i,j) in enumerate(bounds.eqx)
        JEx[i,j] = 1
    end
    JEc = view5(J, eqc, :)
    Jod = zeros(eltype(JEx), size(JEc, 1), size(JEx, 1))
    state.Hf = [Hp -JEx' -JEc';
                -JEx zeros(eltype(JEx), size(JEx,1), size(JEx,1)) Jod';
                -JEc Jod zeros(eltype(JEc), size(JEc,1), size(JEc,1))]
    # Also form the total gradient
    bgrad = state.bgrad
    gI = state.g + JI'*Diagonal(bounds.σc)*(bgrad.slack_c - μ*Sinv2*bgrad.λc)
    for (i,j) in enumerate(ineqx)
        gI[j] += bounds.σx[i]*(bgrad.slack_x[i] - μ*bgrad.λx[i]/sx[i]^2)
    end
    state.gf = [gI;
                bgrad.λxE;
                bgrad.λcE]
    state
end

function update_state!{T}(d, constraints::TwiceDifferentiableConstraintsFunction, state::IPNewtonState{T}, method::IPNewton)
    state.f_x_previous = state.f_x
    bstate, bstep, bounds = state.bstate, state.bstep, constraints.bounds
    state, dslackc = solve_step!(state, constraints)
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
    αmax = convert(eltype(bstate), Inf)
    αmax = estimate_maxstep(αmax, bstate.slack_x, bstep.slack_x)
    αmax = estimate_maxstep(αmax, bstate.slack_c, bstep.slack_c)
    αmax = estimate_maxstep(αmax,
                            view(state.x, bounds.iz).*bounds.σz,
                            view(state.s, bounds.iz).*bounds.σz)

    # Determine the actual distance of movement along the search line
    ϕ = α->lagrangian_linefunc!(α, d, constraints, state, method, dslackc)
    state.alpha, f_update, g_update =
        method.linesearch!(ϕ, T(1), αmax, qp)
    state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    ls_update!(state.x, state.x, state.s, state.alpha)
    ls_update!(bstate, state.constr_c, bstate, bstep, state.alpha, constraints, state, dslackc)

    # Evaluate the constraints at the new position
#    constraints.c!(state.x, state.constr_c)  # already done in ls_update!
    constraints.jacobian!(state.x, state.constr_J)

    # Test for active inequalities, solve immediately for the corresponding s and λ
    # solve_active_inequalities!(d, constraints, state)

    false
end

function solve_step!(state::IPNewtonState, constraints)
    # Solve the Newton step
    local step
    try
        step = -(state.Hf\state.gf)  # do *not* force posdef
    catch
        step = -(svdfact(state.Hf)\state.gf)
    end
    x, s, μ, bounds = state.x, state.s, state.μ, constraints.bounds
    bstate, bstep, bgrad = state.bstate, state.bstep, state.bgrad
    k = unpack_vec!(s, step, 0)
    k = unpack_vec!(bstep.λxE, step, k)
    k = unpack_vec!(bstep.λcE, step, k)
    k == length(step) || error("exhausted targets before step")
    # Solve for the slack variable and λI updates
    # These are only used to estimate αmax, otherwise these are updated by exact formulas
    for (i, j) in enumerate(bounds.ineqx)
        bstep.slack_x[i] = -bgrad.λx[i] + bounds.σx[i]*s[j]
        bstep.λx[i] = -bgrad.slack_x[i] - μ*bstep.slack_x[i]/bstate.slack_x[i]^2
    end
    JI = view5(state.constr_J, bounds.ineqc, :)
    dslackc = Diagonal(bounds.σc)*JI*s
    bstep.slack_c[:] = -bgrad.λc + dslackc
    for i = 1:length(bstep.λc)
        bstep.λc[i] = -bgrad.slack_c[i] - μ*bstep.slack_c[i]/bstate.slack_c[i]^2
    end
    state.stepf = step
    state, dslackc
end

function is_smaller_eps(ref, step)
    ise = true
    for (r, s) in zip(ref, step)
        ise &= (s == 0) | (abs(s) < eps(r))
    end
    ise
end

function quadratic_parameters(bounds::ConstraintBounds, state::IPNewtonState)
    slope = dot(state.stepf, state.gf)
    # For the curvature, use the original hessian (before forcing
    # positive-definiteness)
    q = dot(state.s, state.H*state.s)
    JE = view5(state.constr_J, bounds.eqc, :)
    q -= 2*dot(state.s[bounds.eqx], state.bstep.λxE) + 2*dot(state.s, JE'*state.bstep.λcE)
    state.L, slope, q
end
