immutable IPNewton <: IPOptimizer
    linesearch!::Function
end

IPNewton(; linesearch!::Function = backtrack_constrained!) =
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
    bstate::BarrierStateVars{T}   # value of slack and λ variables (current "position")
    bgrad::BarrierStateVars{T}    # gradient of slack and λ variables at current "position"
    constr_c::Vector{T}   # value of the user-supplied constraints at x
    constr_J::Matrix{T}   # value of the user-supplied Jacobian at x
    @add_linesearch_fields()
    b_ls::BarrierLineSearch{T}
    gf::Vector{T}
    Hf::Matrix{T}
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
    constraints.jacobian!(initial_x, constr_J)
    μ = T(1)
    bstate = BarrierStateVars(constraints.bounds, initial_x, constr_c)
    bgrad = similar(bstate)
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
        bstate,
        bgrad,
        constr_c,
        constr_J,
        @initial_linesearch()..., # Maintain a cache for line search results in state.lsr
        b_ls,
        gf,
        Hf)
    #    μ = initialize_μ_λ!(λv, λc, constraints, initial_x, g, constr_c, constr_J)
    update_g!(d, constraints, state, method)
    update_h!(d, constraints, state, method)
end

function update_g!(d, constraints::TwiceDifferentiableConstraintsFunction, state, method::IPNewton)
    lagrangian_g!(state.g, state.bgrad, d, constraints.bounds, state.x, state.constr_c, state.constr_J, state.bstate, state.μ, method)
end

function update_h!(d, constraints::TwiceDifferentiableConstraintsFunction, state, method::IPNewton)
    μ, Hxx, J = state.μ, state.H, state.constr_J
    d.h!(state.x, Hxx)
    # Collect the values of the coefficients of the inequality constraints
    bounds = constraints.bounds
    ineqc, σc, λc = bounds.ineqc, bounds.σc, state.bstate.λc
    m, n = size(J, 1), size(J, 2)
    λ = zeros(eltype(bounds), m)
    for i = 1:length(ineqc)
        λ[ineqc[i]] -= λc[i]*σc[i]
    end
    # Add the weighted hessian terms from the nonlinear constraints
    constraints.h!(state.x, λ, Hxx)
    # Add the Jacobian terms
    JI = view5(J, ineqc, :)
    Sinv2 = Diagonal(1./state.bstate.slack_c.^2)
    HJ = JI'*Sinv2*JI
    for j = 1:n, i = 1:n
        Hxx[i,j] += μ*HJ[i,j]
    end
    # Add the variable inequalities
    iz, x = bounds.iz, state.x
    for i in iz
        Hxx[i,i] += μ/x[i]^2
    end
    ineqx, sx = bounds.ineqx, state.bstate.slack_x
    for (i,j) in enumerate(ineqx)
        Hxx[j,j] += μ/sx[i]^2
    end
    # Perform a positive factorization
    Hpc, state.Hd = ldltfact(Positive, Hxx)
    Hp = full(Hpc)
    # Now add the equality constraint hessian terms
    eqc, λcE = bounds.eqc, state.bstate.λcE
    fill!(λ, 0)
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
    gI = state.g + JI'*Diagonal(σc)*(bgrad.slack_c - μ*Sinv2*bgrad.λc)
    for (i,j) in enumerate(ineqx)
        gI[j] += bounds.σx[i]*(bgrad.slack_x[i] - μ*bgrad.λx[i]/sx[i]^2)
    end
    state.gf = [gI;
                bgrad.λxE;
                bgrad.λcE]
    state
end
