# TODO: when Optim supports sparse arrays, make a SparseMatrixCSC version of jacobianx


abstract type AbstractBarrierState end

# These are used not only for the current state, but also for the step and the gradient
struct BarrierStateVars{T}
    slack_x::Vector{T}     # values of slack variables for x
    slack_c::Vector{T}     # values of slack variables for c
    λx::Vector{T}          # λ for equality constraints on slack_x
    λc::Vector{T}          # λ for equality constraints on slack_c
    λxE::Vector{T}         # λ for equality constraints on x
    λcE::Vector{T}         # λ for linear/nonlinear equality constraints
end
# Note on λxE:
# We could just set equality-constrained variables to their
# constraint values at the beginning of optimization, but this
# might make the initial guess infeasible in terms of its
# inequality constraints. This would be a much bigger problem than
# not matching the equality constraints.  So we allow them to
# differ, and require that the algorithm can cope with it.

function BarrierStateVars{T}(bounds::ConstraintBounds) where T
    slack_x = Array{T}(undef, length(bounds.ineqx))
    slack_c = Array{T}(undef, length(bounds.ineqc))
    λx = similar(slack_x)
    λc = similar(slack_c)
    λxE = Array{T}(undef, length(bounds.eqx))
    λcE = Array{T}(undef, length(bounds.eqc))
    sv = BarrierStateVars{T}(slack_x, slack_c, λx, λc, λxE, λcE)
end
BarrierStateVars(bounds::ConstraintBounds{T}) where T = BarrierStateVars{T}(bounds)

function BarrierStateVars(bounds::ConstraintBounds{T}, x) where T
    sv = BarrierStateVars(bounds)
    setslack!(sv.slack_x, x, bounds.ineqx, bounds.σx, bounds.bx)
    sv
end
function BarrierStateVars(bounds::ConstraintBounds{T}, x, c) where T
    sv = BarrierStateVars(bounds)
    setslack!(sv.slack_x, x, bounds.ineqx, bounds.σx, bounds.bx)
    setslack!(sv.slack_c, c, bounds.ineqc, bounds.σc, bounds.bc)
    sv
end
function setslack!(slack, v, ineq, σ, b)
    for i = 1:length(ineq)
        dv = v[ineq[i]]-b[i]
        slack[i] = abs(σ[i]*dv)
    end
    slack
end

slack(bstate::BarrierStateVars) = [bstate.slack_x; bstate.slack_c]
lambdaI(bstate::BarrierStateVars) = [bstate.λx; bstate.λc]
lambdaE(bstate::BarrierStateVars) = [bstate.λxE; bstate.λcE] # TODO: Not used by IPNewton?
lambdaI(state::AbstractBarrierState) = lambdaI(state.bstate)
lambdaE(state::AbstractBarrierState) = lambdaE(state.bstate) # TODO: Not used by IPNewton?

Base.similar(bstate::BarrierStateVars) =
    BarrierStateVars(similar(bstate.slack_x),
                     similar(bstate.slack_c),
                     similar(bstate.λx),
                     similar(bstate.λc),
                     similar(bstate.λxE),
                     similar(bstate.λcE))

Base.copy(bstate::BarrierStateVars) =
    BarrierStateVars(copy(bstate.slack_x),
                     copy(bstate.slack_c),
                     copy(bstate.λx),
                     copy(bstate.λc),
                     copy(bstate.λxE),
                     copy(bstate.λcE))

function Base.fill!(b::BarrierStateVars, val)
    fill!(b.slack_x, val)
    fill!(b.slack_c, val)
    fill!(b.λx, val)
    fill!(b.λc, val)
    fill!(b.λxE, val)
    fill!(b.λcE, val)
    b
end

Base.convert(::Type{BarrierStateVars{T}}, bstate::BarrierStateVars) where T =
    BarrierStateVars(convert(Array{T}, bstate.slack_x),
                     convert(Array{T}, bstate.slack_c),
                     convert(Array{T}, bstate.λx),
                     convert(Array{T}, bstate.λc),
                     convert(Array{T}, bstate.λxE),
                     convert(Array{T}, bstate.λcE))

Base.isempty(bstate::BarrierStateVars) = isempty(bstate.slack_x) &
    isempty(bstate.slack_c) & isempty(bstate.λxE) & isempty(bstate.λcE)

Base.eltype(::Type{BarrierStateVars{T}}) where T = T
Base.eltype(sv::BarrierStateVars) = eltype(typeof(sv))

function Base.show(io::IO, b::BarrierStateVars)
    print(io, "BarrierStateVars{$(eltype(b))}:")
    for fn in (:slack_x, :slack_c, :λx, :λc, :λxE, :λcE)
        print(io, "\n  $fn: ")
        show(io, getfield(b, fn))
    end
end

Base.:(==)(v::BarrierStateVars, w::BarrierStateVars) =
    v.slack_x == w.slack_x &&
    v.slack_c == w.slack_c &&
    v.λx == w.λx &&
    v.λc == w.λc &&
    v.λxE == w.λxE &&
    v.λcE == w.λcE

const bsv_seed = sizeof(UInt) == 64 ? 0x145b788192d1cde3 : 0x766a2810
Base.hash(b::BarrierStateVars, u::UInt) =
    hash(b.λcE, hash(b.λxE, hash(b.λc, hash(b.λx, hash(b.slack_c, hash(b.slack_x, u+bsv_seed))))))

function dot(v::BarrierStateVars, w::BarrierStateVars)
    dot(v.slack_x,w.slack_x) +
        dot(v.slack_c, w.slack_c) +
        dot(v.λx, w.λx) +
        dot(v.λc, w.λc) +
        dot(v.λxE, w.λxE) +
        dot(v.λcE, w.λcE)
end

function norm(b::BarrierStateVars, p::Real)
    norm(b.slack_x, p) + norm(b.slack_c, p) +
        norm(b.λx, p) + norm(b.λc, p) +
        norm(b.λxE, p) + norm(b.λcE, p)
end

"""
    BarrierLineSearch{T}

Parameters for interior-point line search methods that use only the value
"""
struct BarrierLineSearch{T}
    c::Vector{T}                  # value of constraints-functions at trial point
    bstate::BarrierStateVars{T}   # trial point for slack and λ variables
end
Base.convert(::Type{BarrierLineSearch{T}}, bsl::BarrierLineSearch) where T =
    BarrierLineSearch(convert(Vector{T}, bsl.c),
                      convert(BarrierStateVars{T}, bsl.bstate))

"""
    BarrierLineSearchGrad{T}

Parameters for interior-point line search methods that exploit the slope.
"""
struct BarrierLineSearchGrad{T}
    c::Vector{T}                  # value of constraints-functions at trial point
    J::Matrix{T}                  # constraints-Jacobian at trial point
    bstate::BarrierStateVars{T}   # trial point for slack and λ variables
    bgrad::BarrierStateVars{T}    # trial point's gradient
end
Base.convert(::Type{BarrierLineSearchGrad{T}}, bsl::BarrierLineSearchGrad) where T =
    BarrierLineSearchGrad(convert(Vector{T}, bsl.c),
                          convert(Matrix{T}, bsl.J),
                          convert(BarrierStateVars{T}, bsl.bstate),
                          convert(BarrierStateVars{T}, bsl.bgrad))

function ls_update!(out::BarrierStateVars, base::BarrierStateVars, step::BarrierStateVars, αs::NTuple{4,Number})
    ls_update!(out.slack_x, base.slack_x, step.slack_x, αs[2])
    ls_update!(out.slack_c, base.slack_c, step.slack_c, αs[2])
    ls_update!(out.λx, base.λx, step.λx, αs[3])
    ls_update!(out.λc, base.λc, step.λc, αs[3])
    ls_update!(out.λxE, base.λxE, step.λxE, αs[4])
    ls_update!(out.λcE, base.λcE, step.λcE, αs[4])
    out
end
ls_update!(out::BarrierStateVars, base::BarrierStateVars, step::BarrierStateVars, αs::Tuple{Number,Number}) =
    ls_update!(out, base, step, (αs[1],αs[1],αs[2],αs[1]))
ls_update!(out::BarrierStateVars, base::BarrierStateVars, step::BarrierStateVars, α::Number) =
    ls_update!(out, base, step, (α,α,α,α))
ls_update!(out::BarrierStateVars, base::BarrierStateVars, step::BarrierStateVars, αs::AbstractVector) =
    ls_update!(out, base, step, αs[1]) # (αs...,))

function initial_convergence(d, state, method::ConstrainedOptimizer, initial_x, options)
    # TODO: Make sure state.bgrad has been evaluated at initial_x
    # state.bgrad normally comes from constraints.c!(..., initial_x) in initial_state
    gradient!(d, initial_x)
    norm(gradient(d), Inf) + norm(state.bgrad, Inf) < options.g_abstol
end

function optimize(d::AbstractObjective, constraints::AbstractConstraints, initial_x::AbstractArray, method::ConstrainedOptimizer,
                  options::Options = Options(;default_options(method)...),
                  state = initial_state(method, options, d, constraints, initial_x))
    #== TODO:
    Let's try to unify this with the unconstrained `optimize` in Optim
    The only thing we'd have to deal with is to dispatch
    the univariate `optimize` to one with empty constraints::AbstractConstraints
    ==#

    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

    tr = OptimizationTrace{typeof(value(d)), typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased, counter_f_tol = false, false, false, 0

    g_converged = initial_convergence(d, state, method, initial_x, options)
    converged = g_converged

    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0

    options.show_trace && print_header(method)
    trace!(tr, d, state, iteration, method, options, t0)

    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        update_state!(d, constraints, state, method, options) && break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS or linesearch errors)

        update_fg!(d, constraints, state, method)

        # TODO: Do we need to rethink f_increased for `ConstrainedOptimizer`s?
        x_converged, f_converged,
        g_converged, converged, f_increased = assess_convergence(state, d, options)
        # With equality constraints, optimization is not necessarily
        # monotonic in the value of the function. If the function
        # change is approximately canceled by a change in the equality
        # violation, it's possible to spuriously satisfy the f_tol
        # criterion. Consequently, we require that the f_tol condition
        # be satisfied a certain number of times in a row before
        # declaring convergence.
        counter_f_tol = f_converged ? counter_f_tol+1 : 0
        converged = converged | (counter_f_tol > options.successive_f_tol)

        # We don't use the Hessian for anything if we have declared convergence,
        # so we might as well not make the (expensive) update if converged == true
        !converged && update_h!(d, constraints, state, method)

        if tracing
            # update trace; callbacks can stop routine early by returning true
            stopped_by_callback = trace!(tr, d, state, iteration, method, options)
        end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        stopped_by_time_limit = time()-t0 > options.time_limit ? true : false
        f_limit_reached = options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
        g_limit_reached = options.g_calls_limit > 0 && g_calls(d) >= options.g_calls_limit ? true : false
        h_limit_reached = options.h_calls_limit > 0 && h_calls(d) >= options.h_calls_limit ? true : false

        if (f_increased && !options.allow_f_increases) || stopped_by_callback ||
            stopped_by_time_limit || f_limit_reached || g_limit_reached || h_limit_reached
            stopped = true
        end
    end # while

    after_while!(d, constraints, state, method, options)

    # we can just check minimum, as we've earlier enforced same types/eltypes
    # in variables besides the option settings
    T = typeof(options.f_reltol)
    Tf = typeof(value(d))
    f_incr_pick = f_increased && !options.allow_f_increases
    return MultivariateOptimizationResults(method,
                                        initial_x,
                                        pick_best_x(f_incr_pick, state),
                                        pick_best_f(f_incr_pick, state, d),
                                        iteration,
                                        iteration == options.iterations,
                                        x_converged,
                                        T(options.x_abstol),
                                        T(options.x_reltol),
                                        x_abschange(state),
                                        x_relchange(state),
                                        f_converged,
                                        T(options.f_abstol),
                                        T(options.f_reltol),
                                        f_abschange(d, state),
                                        f_relchange(d, state),
                                        g_converged,
                                        T(options.g_abstol),
                                        g_residual(d),
                                        f_increased,
                                        tr,
                                        f_calls(d),
                                        g_calls(d),
                                        h_calls(d),
                                        nothing)
end

# Fallbacks (for methods that don't need these)
after_while!(d, constraints::AbstractConstraints, state, method, options) = nothing
update_h!(d, constraints::AbstractConstraints, state, method) = nothing

"""
    initialize_μ_λ!(state, bounds, μ0=:auto, β=0.01)
    initialize_μ_λ!(state, bounds, (Hobj,HcI), μ0=:auto, β=0.01)

Pick μ and λ to ensure that the equality constraints are satisfied
locally (at the current `state.x`), and that the initial gradient
including the barrier would be a descent direction for the problem
without the barrier (μ = 0). This ensures that the search isn't pushed
out of the basin of the user-supplied initial guess.

Upon entry, the objective function gradient, constraint values, and
constraint jacobian must be set in `state.g`, `state.c`, and `state.J`
respectively. If you also wish to ensure that the projection of
Hessian is minimally-perturbed along the initial gradient, supply the
hessian of the objective (`Hobj`) and

    HcI = ∑_i (σ_i/s_i)∇∇ c_{Ii}

for the constraints. This can be obtained as

    HcI = hessianI(state.x, constraints, 1./state.slack_c)

You can manually specify `μ` by supplying a numerical value for
`μ0`. Whether calculated algorithmically or specified manually, the
values of `λ` are set using the chosen `μ`.
"""
function initialize_μ_λ!(state, bounds::ConstraintBounds, Hinfo, μ0::Union{Symbol,Number}, β::Number=1//100)
    if nconstraints(bounds) == 0 && nconstraints_x(bounds) == 0
        state.μ = 0
        fill!(state.bstate, 0)
        return state
    end
    gf = state.g  # must be pre-set to ∇f
    # Calculate projection of ∇f into the subspace spanned by the
    # equality constraint Jacobian
    JE = jacobianE(state, bounds)
    # QRF = qrfact(JE)
    # Q = QRF[:Q]
    # PEg = Q'*(Q*gf)   # in the subspace of JE
    C = JE*JE'
    Cc = cholesky(Positive, C)
    Pperpg = gf-JE'*(Cc \ (JE*gf))   # in the nullspace of JE
    # Set μ
    JI = jacobianI(state, bounds)
    if μ0 == :auto
        # Calculate projections of the Lagrangian's gradient, and
        # possibly hessian, along (∇f)_⟂
        Dperp = dot(Pperpg, Pperpg)
        σ, s = sigma(bounds), slack(state)
        σdivs = σ./s
        Δg = JI'*σdivs
        PperpΔg = Δg - JE'*(Cc \ (JE*Δg))
        DI = dot(PperpΔg, PperpΔg)
        κperp, κI = hessian_projections(Hinfo, Pperpg, (JI*Pperpg)./s)
        # Calculate μ and λI
        μ = β * (κperp == 0 ? sqrt(Dperp/DI) : min(sqrt(Dperp/DI), abs(κperp/κI)))
        if !isfinite(μ)
            Δgtilde = JI'*(1 ./ s)
            PperpΔgtilde = Δgtilde - JE'*(Cc \ (JE*Δgtilde))
            DItilde = dot(PperpΔgtilde, PperpΔgtilde)
            μ = β*sqrt(Dperp/DItilde)
        end
        if !isfinite(μ) || μ == 0
            μ = one(μ)
        end
    else
        μ = convert(eltype(state.x), μ0)
    end
    state.μ = μ
    # Set λI
    @. state.bstate.λx = μ / state.bstate.slack_x
    @. state.bstate.λc = μ / state.bstate.slack_c
    # Calculate λE
    λI = lambdaI(state)
    ∇bI = gf - JI'*λI
#    qrregularize!(QRF)  # in case of any 0 eigenvalues
    λE = Cc \ (JE*∇bI) + (cbar(bounds) - cE(state, bounds))/μ
    k = unpack_vec!(state.bstate.λxE, λE, 0)
    k = unpack_vec!(state.bstate.λcE, λE, k)
    k == length(λE) || error("Something is wrong when initializing μ and λ.")
    state
end
function initialize_μ_λ!(state, bounds::ConstraintBounds, μ0::Union{Number,Symbol}, β::Number=1//100)
    initialize_μ_λ!(state, bounds, nothing, μ0, β)
end

function hessian_projections(Hinfo::Tuple{AbstractMatrix,AbstractMatrix}, Pperpg, y)
    κperp = dot(Hinfo[1]*Pperpg, Pperpg)
    κI = dot(Hinfo[2]*Pperpg, Pperpg) + dot(y,y)
    κperp, κI
end
hessian_projections(Hinfo::Nothing, Pperpg::AbstractVector{T}) where T = convert(T, Inf), zero(T)

function jacobianE(state, bounds::ConstraintBounds)
    J, x = state.constr_J, state.x
    JEx = jacobianx(J, bounds.eqx)
    JEc = view(J, bounds.eqc, :)
    JE = vcat(JEx, JEc)
end
jacobianE(state, constraints) = jacobianE(state, constraints.bounds)

function jacobianI(state, bounds::ConstraintBounds)
    J, x = state.constr_J, state.x
    JIx = jacobianx(J, bounds.ineqx)
    JIc = view(J, bounds.ineqc, :)
    JI = vcat(JIx, JIc)
end
jacobianI(state, constraints) = jacobianI(state, constraints.bounds)

# TODO: when Optim supports sparse arrays, make a SparseMatrixCSC version
function jacobianx(J::AbstractArray, indx)
    Jx = zeros(eltype(J), length(indx), size(J, 2))
    for (i,j) in enumerate(indx)
        Jx[i,j] = 1
    end
    Jx
end

function sigma(bounds::ConstraintBounds)
    [bounds.σx; bounds.σc]  # don't include σz
end
sigma(constraints) = sigma(constraints.bounds)

slack(state) = slack(state.bstate)

cbar(bounds::ConstraintBounds) = [bounds.valx; bounds.valc]
cbar(constraints) = cbar(constraints.bounds)
cE(state, bounds::ConstraintBounds) = [state.x[bounds.eqx]; state.constr_c[bounds.eqc]]

function hessianI!(h, x, constraints, λcI, μ)
    λ = userλ(λcI, constraints)
    constraints.h!(h, x, λ)
    h
end

"""
   hessianI(x, constraints, λcI, μ) -> h

Compute the hessian at `x` of the `λcI`-weighted sum of user-supplied
constraint functions for just the inequalities.  This also includes
contributions from any variables with bounds at 0, since those do not
cause introduction of a slack variable. Other (nonzero) box
constraints do not contribute to `h`, because the hessian of `x_i` is
zero. (They contribute indirectly via their slack variables.)
"""
hessianI(x, constraints, λcI, μ) =
    hessianI!(zeros(eltype(x), length(x), length(x)), x, constraints, λcI, μ)

"""
    userλ(λcI, bounds) -> λ

Accumulates `λcI` into a vector `λ` ordered as the user-supplied
constraint functions `c`. Upper and lower bounds are summed, weighted
by `σ`. The resulting λ includes an overall negative sign so that this
becomes the coefficient for the user-supplied hessian.

This is relevant only for the inequalities. If you want the λ for just
the equalities, you can use `λ[bounds.ceq] = λcE` for a zero-filled `λ`.
"""
function userλ(λcI, bounds::ConstraintBounds)
    ineqc, σc = bounds.ineqc, bounds.σc
    λ = zeros(eltype(bounds), nconstraints(bounds))
    for i = 1:length(ineqc)
        λ[ineqc[i]] -= λcI[i]*σc[i]
    end
    λ
end
userλ(λcI, constraints) = userλ(λcI, constraints.bounds)

## Computation of the Lagrangian and its gradient
# This is in a parametrization that is also useful during linesearch
# TODO: `lagrangian` does not seem to be used (IPNewton)?
function lagrangian(d, bounds::ConstraintBounds, x, c, bstate::BarrierStateVars, μ)
    f_x = NLSolversBase.value!(d, x)
    ev = equality_violation(bounds, x, c, bstate)
    L_xsλ = f_x + barrier_value(bounds, x, bstate, μ) + ev
    f_x, L_xsλ, ev
end

function lagrangian_fg!(gx, bgrad, d, bounds::ConstraintBounds, x, c, J, bstate::BarrierStateVars, μ)
    fill!(bgrad, 0)
    f_x, g_x = NLSolversBase.value_gradient!(d,x)
    gx .= g_x
    ev = equality_violation(bounds, x, c, bstate)
    L_xsλ = f_x + barrier_value(bounds, x, bstate, μ) + ev
    barrier_grad!(bgrad, bounds, x, bstate, μ)
    equality_grad!(gx, bgrad, bounds, x, c, J, bstate)
    f_x, L_xsλ, ev
end

# TODO: do we need lagrangian_vec? Maybe for automatic differentiation?
## Computation of Lagrangian and derivatives when passing all parameters as a single vector
function lagrangian_vec(p, d, bounds::ConstraintBounds, x, c::AbstractArray, bstate::BarrierStateVars, μ)
    unpack_vec!(x, bstate, p)
    f_x, L_xsλ, ev = lagrangian(d, bounds, x, c, bstate, μ)
    L_xsλ
end
function lagrangian_vec(p, d, bounds::ConstraintBounds, x, c::Function, bstate::BarrierStateVars, μ)
    # Use this version when using automatic differentiation
    unpack_vec!(x, bstate, p)
    f_x, L_xsλ, ev = lagrangian(d, bounds, x, c(x), bstate, μ)
    L_xsλ
end
function lagrangian_fgvec!(p, storage, gx, bgrad, d, bounds::ConstraintBounds, x, c, J, bstate::BarrierStateVars, μ)
    unpack_vec!(x, bstate, p)
    f_x, L_xsλ, ev = lagrangian_fg!(gx, bgrad, d, bounds, x, c, J, bstate, μ)
    pack_vec!(storage, gx, bgrad)
    L_xsλ
end

## for line searches that don't use the gradient along the line
function lagrangian_linefunc(αs, d, constraints, state)
    _lagrangian_linefunc(αs, d, constraints, state)[2]
end

function _lagrangian_linefunc(αs, d, constraints, state)
    b_ls, bounds = state.b_ls, constraints.bounds
    ls_update!(state.x_ls, state.x, state.s, alphax(αs))
    ls_update!(b_ls.bstate, state.bstate, state.bstep, αs)
    constraints.c!(b_ls.c, state.x_ls)
    lagrangian(d, constraints.bounds, state.x_ls, b_ls.c, b_ls.bstate, state.μ)
end
alphax(α::Number) = α
alphax(αs::Union{Tuple,AbstractVector}) = αs[1]

function lagrangian_linefunc!(α, d, constraints, state, method::IPOptimizer{typeof(backtrack_constrained)})
    # For backtrack_constrained, the last evaluation is the one we
    # keep, so it's safe to store the results in state
    state.f_x, state.L, state.ev = _lagrangian_linefunc(α, d, constraints, state)
    state.L
end
lagrangian_linefunc!(α, d, constraints, state, method) = lagrangian_linefunc(α, d, constraints, state)


## for line searches that do use the gradient along the line
function lagrangian_lineslope(αs, d, constraints, state)
    f_x, L, ev, slope = _lagrangian_lineslope(αs, d, constraints, state)
    L, slope
end

function _lagrangian_lineslope(αs, d, constraints, state)
    b_ls, bounds = state.b_ls, constraints.bounds
    bstep, bgrad = state.bstep, b_ls.bgrad
    ls_update!(state.x_ls, state.x, state.s, alphax(αs))
    ls_update!(b_ls.bstate, state.bstate, bstep, αs)
    constraints.c!(b_ls.c, state.x_ls)
    constraints.jacobian!(b_ls.J, state.x_ls)
    f_x, L, ev = lagrangian_fg!(state.g, bgrad, d, bounds, state.x_ls, b_ls.c, b_ls.J, b_ls.bstate, state.μ)
    slopeα = slopealpha(state.s, state.g, bstep, bgrad)
    f_x, L, ev, slopeα
end

function lagrangian_lineslope!(αs, d, constraints, state, method::IPOptimizer{typeof(backtrack_constrained_grad)})
    # For backtrack_constrained, the last evaluation is the one we
    # keep, so it's safe to store the results in state
    state.f_x, state.L, state.ev, slope = _lagrangian_lineslope(αs, d, constraints, state)
     state.L, slope
end
lagrangian_lineslope!(αs, d, constraints, state, method) = lagrangian_lineslope(αs, d, constraints, state)

slopealpha(sx, gx, bstep, bgrad) = dot(sx, gx) +
    dot(bstep.slack_x, bgrad.slack_x) + dot(bstep.slack_c, bgrad.slack_c) +
    dot(bstep.λx, bgrad.λx) + dot(bstep.λc, bgrad.λc) +
    dot(bstep.λxE, bgrad.λxE) + dot(bstep.λcE, bgrad.λcE)

function linesearch_anon(d, constraints, state, method::IPOptimizer{typeof(backtrack_constrained_grad)})
    αs->lagrangian_lineslope!(αs, d, constraints, state, method)
end
function linesearch_anon(d, constraints, state, method::IPOptimizer{typeof(backtrack_constrained)})
    αs->lagrangian_linefunc!(αs, d, constraints, state, method)
end

## Computation of Lagrangian terms: barrier penalty
"""
    barrier_value(constraints, state) -> val
    barrier_value(bounds, x, sx, sc, μ) -> val

Compute the value of the barrier penalty at the current `state`, or at
a position (`x`,`sx`,`sc`), where `x` is the current position, `sx`
are the coordinate slack variables, and `sc` are the linear/nonlinear
slack variables. `bounds` holds the parsed bounds.
"""
function barrier_value(bounds::ConstraintBounds, x, sx, sc, μ)
    # bμ is the coefficient of μ in the barrier penalty
    bμ = _bv(sx) +  # coords with other bounds
         _bv(sc)    # linear/nonlinear constr.
    μ*bμ
end
barrier_value(bounds::ConstraintBounds, x, bstate::BarrierStateVars, μ) =
    barrier_value(bounds, x, bstate.slack_x, bstate.slack_c, μ)
barrier_value(bounds::ConstraintBounds, state) =
    barrier_value(bounds, state.x, state.bstate.slack_x, state.bstate.slack_c, state.μ)
barrier_value(constraints::AbstractConstraints, state) =
    barrier_value(constraints.bounds, state)

# don'tcall this barrier_value because it lacks μ
function _bv(v, idx, σ) # TODO: Not used, delete? (IPNewton)
    ret = loginf(one(eltype(σ))*one(eltype(v)))
    for (i,iv) in enumerate(idx)
        ret += loginf(σ[i]*v[iv])
    end
    -ret
end

_bv(v) = isempty(v) ? loginf(one(eltype(v))) : -sum(loginf, v)

loginf(δ) = δ > 0 ? log(δ) : -oftype(δ, Inf)

"""
    barrier_grad!(bgrad, bounds, x, bstate, μ)
    barrier_grad!(gsx, gsc, bounds, x, sx, sc, μ)

Compute the gradient of the barrier penalty at (`x`,`sx`,`sc`), where
`x` is the current position, `sx` are the coordinate slack variables,
and `sc` are the linear/nonlinear slack
variables. `bounds::ConstraintBounds` holds the parsed bounds.

The result is *added* to `gsx`, and `gsc`, so these vectors
need to be initialized appropriately.
"""
function barrier_grad!(gsx, gsc, bounds::ConstraintBounds, x, sx, sc, μ)
    barrier_grad!(gsx, sx, μ)
    barrier_grad!(gsc, sc, μ)
    nothing
end
barrier_grad!(bgrad, bounds::ConstraintBounds, x, bstate, μ) =
    barrier_grad!(bgrad.slack_x, bgrad.slack_c, bounds, x, bstate.slack_x, bstate.slack_c, μ)

function barrier_grad!(out, v, μ)
    for i = 1:length(out)
        out[i] -= μ/v[i]
    end
    nothing
end


## Computation of Lagrangian terms: equality constraints penalty

"""
    equality_violation([f=identity], bounds, x, c, bstate) -> val
    equality_violation([f=identity], bounds, x, c, sx, sc, λx, λc, λxE, λcE) -> val

Compute the sum of `f(v_i)`, where `v_i = λ_i*(target - observed)`
measures the difference between the current state and the
equality-constrained state. `bounds::ConstraintBounds` holds the
parsed bounds. `x` is the current position, `sx` are the coordinate
slack variables, and `sc` are the linear/nonlinear slack
variables. `c` holds the values of the linear-nonlinear constraints,
and the λ arguments hold the Lagrange multipliers for `x`, `sx`, `sc`, and
`c` respectively.
"""
function equality_violation(f, bounds::ConstraintBounds, x, c, sx, sc, λx, λc, λxE, λcE)
    ev = equality_violation(f, sx, x, bounds.ineqx, bounds.σx, bounds.bx, λx) +
         equality_violation(f, sc, c, bounds.ineqc, bounds.σc, bounds.bc, λc) +
         equality_violation(f, x, bounds.valx, bounds.eqx, λxE) +
         equality_violation(f, c, bounds.valc, bounds.eqc, λcE)
end
equality_violation(bounds::ConstraintBounds, x, c, sx, sc, λx, λc, λxE, λcE) =
    equality_violation(identity, bounds, x, c, sx, sc, λx, λc, λxE, λcE)
function equality_violation(f, bounds::ConstraintBounds, x, c, bstate::BarrierStateVars)
    equality_violation(f, bounds, x, c, bstate.slack_x, bstate.slack_c,
                       bstate.λx, bstate.λc, bstate.λxE, bstate.λcE)
end
equality_violation(bounds::ConstraintBounds, x, c, bstate::BarrierStateVars) =
    equality_violation(identity, bounds, x, c, bstate)
equality_violation(f, bounds::ConstraintBounds, state::AbstractBarrierState) =
    equality_violation(f, bounds, state.x, state.constr_c, state.bstate)
equality_violation(bounds::ConstraintBounds, state::AbstractBarrierState) =
    equality_violation(identity, bounds, state)
equality_violation(f, constraints::AbstractConstraints, state::AbstractBarrierState) =
    equality_violation(f, constraints.bounds, state)
equality_violation(constraints::AbstractConstraints, state::AbstractBarrierState) =
    equality_violation(constraints.bounds, state)

# violations of s = σ*(v-b)
function equality_violation(f, s, v, ineq, σ, b, λ)
    ret = f(zero(eltype(λ))*(zero(eltype(s))-zero(eltype(σ))*(zero(eltype(v))-zero(eltype(b)))))
    for (i,iv) in enumerate(ineq)
        ret += f(λ[i]*(s[i] - σ[i]*(v[iv]-b[i])))
    end
    ret
end

# violations of v = target
function equality_violation(f, v, target, idx, λ)
    ret = f(zero(eltype(λ))*(zero(eltype(v))-zero(eltype(target))))
    for (i,iv) in enumerate(idx)
        ret += f(λ[i]*(target[i] - v[iv]))
    end
    ret
end

"""
    equality_grad!(gx, gbstate, bounds, x, c, J, bstate)

Compute the gradient of `equality_violation`, storing the result in `gx` (an array) and `gbstate::BarrierStateVars`.
"""
function equality_grad!(gx, gsx, gsc, gλx, gλc, gλxE, gλcE, bounds::ConstraintBounds, x, c, J, sx, sc, λx, λc, λxE, λcE)
    equality_grad_var!(gsx, gx, bounds.ineqx, bounds.σx, λx)
    equality_grad_var!(gsc, gx, bounds.ineqc, bounds.σc, λc, J)
    gx[bounds.eqx] .= gx[bounds.eqx] .- λxE
    equality_grad_var!(gx, bounds.eqc, λcE, J)
    equality_grad_λ!(gλx, sx, x, bounds.ineqx, bounds.σx, bounds.bx)
    equality_grad_λ!(gλc, sc, c, bounds.ineqc, bounds.σc, bounds.bc)
    equality_grad_λ!(gλxE, x, bounds.valx, bounds.eqx)
    equality_grad_λ!(gλcE, c, bounds.valc, bounds.eqc)
end
equality_grad!(gx, gb::BarrierStateVars, bounds::ConstraintBounds, x, c, J, b::BarrierStateVars) =
    equality_grad!(gx, gb.slack_x, gb.slack_c, gb.λx, gb.λc, gb.λxE, gb.λcE,
                   bounds, x, c, J,
                   b.slack_x, b.slack_c, b.λx, b.λc, b.λxE, b.λcE)

# violations of s = σ*(x-b)
function equality_grad_var!(gs, gx, ineq, σ, λ)
    for (i,ix) in enumerate(ineq)
        λi = λ[i]
        gs[i] += λi
        gx[ix] -= λi*σ[i]
    end
    nothing
end

function equality_grad_var!(gs, gx, ineq, σ, λ, J)
    @. gs = gs + λ
    if !isempty(ineq)
        gx .= gx .- view(J, ineq, :)'*(λ.*σ)
    end
    nothing
end

function equality_grad_λ!(gλ, s, v, ineq, σ, b)
    for (i,iv) in enumerate(ineq)
        gλ[i] += s[i] - σ[i]*(v[iv]-b[i])
    end
    nothing
end

# violations of v = target
function equality_grad_var!(gx, idx, λ, J)
    if !isempty(idx)
        gx .= gx .- view(J, idx, :)'*λ
    end
    nothing
end

function equality_grad_λ!(gλ, v, target, idx)
    for (i,iv) in enumerate(idx)
        gλ[i] += target[i] - v[iv]
    end
    nothing
end

"""
    isfeasible(constraints, state) -> Bool
    isfeasible(constraints, x, c) -> Bool
    isfeasible(constraints, x) -> Bool
    isfeasible(bounds, x, c) -> Bool

Return `true` if point `x` is feasible, given the `constraints` which
specify bounds `lx`, `ux`, `lc`, and `uc`. `x` is feasible if

    lx[i] <= x[i] <= ux[i]
    lc[i] <= c[i] <= uc[i]

for all possible `i`.
"""
function isfeasible(bounds::ConstraintBounds, x, c)
    isf = true
    for (i,j) in enumerate(bounds.eqx)
        isf &= x[j] == bounds.valx[i]
    end
    for (i,j) in enumerate(bounds.ineqx)
        isf &= bounds.σx[i]*(x[j] - bounds.bx[i]) >= 0
    end
    for (i,j) in enumerate(bounds.eqc)
        isf &= c[j] == bounds.valc[i]
    end
    for (i,j) in enumerate(bounds.ineqc)
        isf &= bounds.σc[i]*(c[j] - bounds.bc[i]) >= 0
    end
    isf
end
isfeasible(constraints, state::AbstractBarrierState) = isfeasible(constraints, state.x, state.constraints_c)
function isfeasible(constraints, x)
    # don't assume c! returns c (which means this is a little more awkward)
    c = Array{eltype(x)}(undef, constraints.bounds.nc)
    constraints.c!(c, x)
    isfeasible(constraints, x, c)
end
isfeasible(constraints::AbstractConstraints, x, c) = isfeasible(constraints.bounds, x, c)
isfeasible(constraints::Nothing, state::AbstractBarrierState) = true
isfeasible(constraints::Nothing, x) = true

"""
    isinterior(constraints, state) -> Bool
    isinterior(constraints, x, c) -> Bool
    isinterior(constraints, x) -> Bool
    isinterior(bounds, x, c) -> Bool

Return `true` if point `x` is on the interior of the allowed region,
given the `constraints` which specify bounds `lx`, `ux`, `lc`, and
`uc`. `x` is in the interior if

    lx[i] < x[i] < ux[i]
    lc[i] < c[i] < uc[i]

for all possible `i`.
"""
function isinterior(bounds::ConstraintBounds, x, c)
    isi = true
    for (i,j) in enumerate(bounds.ineqx)
        isi &= bounds.σx[i]*(x[j] - bounds.bx[i]) > 0
    end
    for (i,j) in enumerate(bounds.ineqc)
        isi &= bounds.σc[i]*(c[j] - bounds.bc[i]) > 0
    end
    isi
end
isinterior(constraints, state::AbstractBarrierState) = isinterior(constraints, state.x, state.constraints_c)
function isinterior(constraints, x)
    c = Array{eltype(x)}(undef, constraints.bounds.nc)
    constraints.c!(c, x)
    isinterior(constraints, x, c)
end
isinterior(constraints::AbstractConstraints, x, c) = isinterior(constraints.bounds, x, c)
isinterior(constraints::Nothing, state::AbstractBarrierState) = true
isinterior(constraints::Nothing, x) = true

## Utilities for representing total state as single vector
# TODO: Most of these seem to be unused (IPNewton)?
function pack_vec(x, b::BarrierStateVars)
    n = length(x)
    for fn in fieldnames(b)
        n += length(getfield(b, fn))
    end
    vec = Array{eltype(x)}(undef, n)
    pack_vec!(vec, x, b)
end

function pack_vec!(vec, x, b::BarrierStateVars)
    k = pack_vec!(vec, x, 0)
    for fn in fieldnames(b)
        k = pack_vec!(vec, getfield(b, fn), k)
    end
    k == length(vec) || throw(DimensionMismatch("vec should have length $k, got $(length(vec))"))
    vec
end
function pack_vec!(vec, x, k::Int)
    for i = 1:length(x)
        vec[k+=1] = x[i]
    end
    k
end
function unpack_vec!(x, b::BarrierStateVars, vec::Vector)
    k = unpack_vec!(x, vec, 0)
    for fn in fieldnames(b)
        k = unpack_vec!(getfield(b, fn), vec, k)
    end
    k == length(vec) || throw(DimensionMismatch("vec should have length $k, got $(length(vec))"))
    x, b
end
function unpack_vec!(x, vec::Vector, k::Int)
    for i = 1:length(x)
        x[i] = vec[k+=1]
    end
    k
end

## More utilities
function estimate_maxstep(αmax, x, s)
    for i = 1:length(s)
        si = s[i]
        if si < 0
            αmax = min(αmax, -x[i]/si)
        end
    end
    αmax
end

# TODO: This is not used anymore??
function qrregularize!(QRF)
    R = QRF[:R]
    for i = 1:size(R, 1)
        if R[i,i] == 0
            R[i,i] = 1
        end
    end
    QRF
end
