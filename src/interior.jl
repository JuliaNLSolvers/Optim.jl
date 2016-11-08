abstract AbstractBarrierState

# These are used not only for the current state, but also for the step and the gradient
immutable BarrierStateVars{T}
    slack_x::Vector{T}     # values of slack variables for x
    slack_c::Vector{T}     # values of slack variables for c
    active_x::Vector{Bool} # active constraints for x (see solve_active_inequalities)
    active_c::Vector{Bool} # active constraints for c
    λxE::Vector{T}         # λ for equality constraints on x
    λx::Vector{T}          # λ for equality constraints on slack_x
    λc::Vector{T}          # λ for equality constraints on slack_c
    λcE::Vector{T}         # λ for linear/nonlinear equality constraints
end
# Note on λxE:
# We could just set equality-constrained variables to their
# constraint values at the beginning of optimization, but this
# might make the initial guess infeasible in terms of its
# inequality constraints. This would be a much bigger problem than
# not matching the equality constraints.  So we allow them to
# differ, and require that the algorithm can cope with it.

@compat function (::Type{BarrierStateVars{T}}){T}(bounds::ConstraintBounds)
    slack_x = Array{T}(length(bounds.ineqx))
    slack_c = Array{T}(length(bounds.ineqc))
    λxE = Array{T}(length(bounds.eqx))
    λx = similar(slack_x)
    λc = similar(slack_c)
    λcE = Array{T}(length(bounds.eqc))
    sv = BarrierStateVars{T}(slack_x, slack_c, fill(false, length(slack_x)),
                             fill(false, length(slack_c)), λxE, λx, λc, λcE)
end
BarrierStateVars{T}(bounds::ConstraintBounds{T}) = BarrierStateVars{T}(bounds)

function BarrierStateVars{T}(bounds::ConstraintBounds{T}, x)
    sv = BarrierStateVars(bounds)
    setslack!(sv.slack_x, sv.active_x, x, bounds.ineqx, bounds.σx, bounds.bx)
    sv
end
function BarrierStateVars{T}(bounds::ConstraintBounds{T}, x, c)
    sv = BarrierStateVars(bounds)
    setslack!(sv.slack_x, sv.active_x, x, bounds.ineqx, bounds.σx, bounds.bx)
    setslack!(sv.slack_c, sv.active_c, c, bounds.ineqc, bounds.σc, bounds.bc)
    sv
end
function setslack!(slack, active, v, ineq, σ, b)
    for i = 1:length(ineq)
        dv = v[ineq[i]]-b[i]
        slack[i] = σ[i]*dv
        active[i] = dv == 0
    end
    slack
end

Base.similar(bstate::BarrierStateVars) =
    BarrierStateVars(similar(bstate.slack_x),
                     similar(bstate.slack_c),
                     similar(bstate.active_x),
                     similar(bstate.active_c),
                     similar(bstate.λxE),
                     similar(bstate.λx),
                     similar(bstate.λc),
                     similar(bstate.λcE))

function Base.fill!(b::BarrierStateVars, val)
    fill!(b.slack_x, val)
    fill!(b.slack_c, val)
    fill!(b.active_x, false)
    fill!(b.active_c, false)
    fill!(b.λxE, val)
    fill!(b.λx, val)
    fill!(b.λc, val)
    fill!(b.λcE, val)
    b
end

Base.eltype{T}(::Type{BarrierStateVars{T}}) = T
Base.eltype(sv::BarrierStateVars) = eltype(typeof(sv))

function Base.show(io::IO, b::BarrierStateVars)
    print(io, "BarrierStateVars{$(eltype(b))}:")
    for fn in fieldnames(b)
        print(io, "\n  $fn: ")
        show(io, getfield(b, fn))
    end
end

@compat Base.:(==)(v::BarrierStateVars, w::BarrierStateVars) =
    v.slack_x == w.slack_x &&
    v.slack_c == w.slack_c &&
    v.λxE == w.λxE &&
    v.λx == w.λx &&
    v.λc == w.λc &&
    v.λcE == w.λcE

const bsv_seed = sizeof(UInt) == 64 ? 0x145b788192d1cde3 : 0x766a2810
Base.hash(b::BarrierStateVars, u::UInt) =
    hash(b.λcE, hash(b.λc, hash(b.λx, hash(b.λxE, hash(b.slack_c, hash(b.slack_x, u+bsv_seed))))))


"""
    BarrierLineSearch{T}

Parameters for interior-point line search methods that use only the value
"""
immutable BarrierLineSearch{T}
    c::Vector{T}                  # value of constraints-functions at trial point
    bstate::BarrierStateVars{T}   # trial point for slack and λ variables
end

"""
    BarrierLineSearchGrad{T}

Parameters for interior-point line search methods that exploit the slope.
"""
immutable BarrierLineSearchGrad{T}
    c::Vector{T}                  # value of constraints-functions at trial point
    J::Matrix{T}                  # constraints-Jacobian at trial point
    bstate::BarrierStateVars{T}   # trial point for slack and λ variables
    bgrad::BarrierStateVars{T}    # trial point's gradient
end

function ls_update!(out::BarrierStateVars, base::BarrierStateVars, step::BarrierStateVars, α)
    ls_update!(out.slack_x, base.slack_x, step.slack_x, α)
    ls_update!(out.slack_c, base.slack_c, step.slack_c, α)
    ls_update!(out.λxE, base.λxE, step.λxE, α)
    ls_update!(out.λx, base.λx, step.λx, α)
    ls_update!(out.λc, base.λc, step.λc, α)
    ls_update!(out.λcE, base.λcE, step.λcE, α)
    out
end

# Explicit solution for slack, λ when an inequality constraint is
# "active." This is necessary (or at least helpful) when c-b == 0 due
# to roundoff error, in which case the KKT equations don't have an
# exact solution within the precision.  We punt on the ∂λ equation
# (which reduces to the slack, which should be small anyway), and
# focus on the ∂x and ∂slack equations (therefore setting slack and
# λ). By setting these to their exact solutions, we blance the forces
# due to the barrier.
function solve_active_inequalities!(d, constraints, state)
    x, c, bstate, bounds = state.x, state.constr_c, state.bstate, constraints.bounds
    nactive, nchanged = tally_active!(bstate.active_x, 0, 0, x, bounds.ineqx, bounds.bx)
    nx = nactive
    nactive, nchanged = tally_active!(bstate.active_c, nactive, nchanged, c, bounds.ineqc, bounds.bc, )
    if nactive == 0 || nchanged == 0
        return nothing
    end
    # Calculate the necessary gradients
    d.g!(state.x, state.g)
    constraints.jacobian!(state.x, state.constr_J)
    # Solve for the Lagrange multipliers
    ic, ix = bounds.ineqc[bstate.active_c], bounds.ineqx[bstate.active_x]
    Jx = view5(state.constr_J, ic, ix)
    Jact = view5(state.constr_J, ic, :)
    Cactive = [eye(eltype(Jx), nx, nx) Jx'; Jx Jact*Jact']
    pactive = [view(state.g, ix); Jact*state.g]
    λactive = (Cactive\pactive).*[bounds.σx[bstate.active_x]; bounds.σc[bstate.active_c]]
    # Set the state
    k = set_active_params!(bstate.slack_x, bstate.λx, bstate.active_x, λactive, state.μ, 0)
    k = set_active_params!(bstate.slack_c, bstate.λc, bstate.active_c, λactive, state.μ, k)
    k == length(λactive) || error("something is wrong")
    nothing
end

function tally_active!(active, nactive, nchanged, c, ineq, b)
    for (i,j) in enumerate(ineq)
        isactive = c[j] == b[i]
        nactive += isactive
        nchanged += isactive != active[i]
        active[i] = isactive
    end
    nactive, nchanged
end

function set_active_params!(slack, λ, active, λtarget, μ, k)
    for i = 1:length(active)
        active[i] || continue
        λk = λtarget[k+=1]
        λ[i] = λk
        slack[i] = μ/λk
    end
    k
end

"""
    initialize_μ_λE!(λxE, λcE, constraints, x, g, constr_c, constr_J, β=0.01) -> μ

Pick μ and λ to ensure that the equality constraints are satisfied
locally, and that the initial gradient including the barrier would be
a descent direction for the problem without the barrier (μ = 0). This
ensures that the search isn't pushed out of the basin of the
user-supplied initial guess.

`λv` and `λc` are the Lagrange multipliers for the variables and extra
(non-variable) constraints; these are pre-allocated storage for the
output, and their input values are not used. `constraints` is an
`AbstractConstraintsFunction`, `x` is the position (must be a feasible
interior point), `g` is the gradient of the objective at `x`, and
`constr_c` and `constr_J` contain the values and Jacobian of the extra
constraints evaluated at `x`. `β` (optional) specifies the fraction of
the objective's gradient that may be diminished by the barrier.

In addition to setting `λxE` and `λcE`, this returns `μ`, the value of
the barrier penalty.
"""
function initialize_μ_λ!(λx, λc, bounds::ConstraintBounds, x, g, c, J, β=1//100)
    length(c) + length(bounds.iz) + length(bounds.ineqx) == 0 && return zero(eltype(x))
    # Calculate the projection matrix
    JEx = zeros(eltype(J), length(bounds.eqx), length(x))
    for (i,j) in enumerate(bounds.eqx)
        JEx[i,j] = 1
    end
    JEc = view5(J, bounds.eqc, :)
    JE = vcat(JEx, JEc)
    CE = JE*JE'
    CEc = cholfact(Positive, CE)
    Pg = g - JE'*(CEc \ (JE*g)) # the projected gradient of the objective (orthog to all == constr.)
    # Calculate the barrier deviation and projection onto inequality normals
    Δb = [x[bounds.iz]; x[bounds.ineqx] - bounds.bx; c[bounds.ineqc] - bounds.bc]
    JIx = zeros(eltype(J), length(bounds.iz)+length(bounds.ineqx), length(x))
    for (i,j) in enumerate([bounds.iz; bounds.ineqx])
        JIx[i,j] = 1
    end
    JIc = view5(J, bounds.ineqc, :)
    JI = vcat(JIx, JIc)
    JIg = JI*Pg
    # Solve for μ
    λtilde = 1./Δb
    μden = dot(λtilde, JIg)
    if μden == 0
        μden = maximum(abs(λtilde).*abs(JIg))*length(Δb)
    end
    μ = β*dot(Pg, Pg)/abs(μden)
    μ = μden != 0 ? μ : oftype(μ, 1)
    # Solve for λE
    gb = g - μ*(JI'*λtilde)
    Pgb = gb - JE'*(CEc \ (JE*gb))
    λE = CEc \ (JE*Pgb)
    k = unpack_vec!(λx, λE, 0)
    k = unpack_vec!(λc, λE, k)
    k == length(λE) || error("something is wrong")
    μ
end
initialize_μ_λ!(λx, λc, constraints::AbstractConstraintsFunction, x, g, c, J, args...) =
    initialize_μ_λ!(λx, λc, constraints.bounds, x, g, c, J, args...)
# Fallbacks (for methods that don't need these)
after_while!(d, constraints::AbstractConstraintsFunction, state, method, options) = nothing
update_h!(d, constraints::AbstractConstraintsFunction, state, method) = nothing

## Computation of the Lagrangian and its gradient
# This is in a parametrization that is also useful during linesearch

function lagrangian(d, bounds::ConstraintBounds, x, c, bstate::BarrierStateVars, μ)
    f_x = d.f(x)
    L_xsλ = f_x + barrier_value(bounds, x, bstate, μ) +
            equality_violation(bounds, x, c, bstate)
    f_x, L_xsλ
end

function lagrangian_g!(gx, bgrad, d, bounds::ConstraintBounds, x, c, J, bstate::BarrierStateVars, μ)
    fill!(bgrad, 0)
    d.g!(x, gx)
    barrier_grad!(gx, bgrad, bounds, x, bstate, μ)
    equality_grad!(gx, bgrad, bounds, x, c, J, bstate)
    nothing
end

function lagrangian_fg!(gx, bgrad, d, bounds::ConstraintBounds, x, c, J, bstate::BarrierStateVars, μ)
    fill!(bgrad, 0)
    f_x = d.fg!(x, gx)
    L_xsλ = f_x + barrier_value(bounds, x, bstate, μ) +
        equality_violation(bounds, x, c, bstate)
    barrier_grad!(gx, bgrad, bounds, x, bstate, μ)
    equality_grad!(gx, bgrad, bounds, x, c, J, bstate)
    f_x, L_xsλ
end

## Computation of Lagrangian and derivatives when passing all parameters as a single vector
function lagrangian_vec(p, d, bounds::ConstraintBounds, x, c::AbstractArray, bstate::BarrierStateVars, μ)
    unpack_vec!(x, bstate, p)
    f_x, L_xsλ = lagrangian(d, bounds, x, c, bstate, μ)
    L_xsλ
end
function lagrangian_vec(p, d, bounds::ConstraintBounds, x, c::Function, bstate::BarrierStateVars, μ)
    # Use this version when using automatic differentiation
    unpack_vec!(x, bstate, p)
    f_x, L_xsλ = lagrangian(d, bounds, x, c(x), bstate, μ)
    L_xsλ
end
function lagrangian_fgvec!(p, storage, gx, bgrad, d, bounds::ConstraintBounds, x, c, J, bstate::BarrierStateVars, μ)
    unpack_vec!(x, bstate, p)
    f_x, L_xsλ = lagrangian_fg!(gx, bgrad, d, bounds, x, c, J, bstate, μ)
    pack_vec!(storage, gx, bgrad)
    L_xsλ
end

# for line searches that don't use the gradient along the line
function lagrangian_linefunc(α, d, constraints, state)
    b_ls = state.b_ls
    ls_update!(state.x_ls, state.x, state.s, α)
    ls_update!(b_ls.bstate, state.bstate, state.bstep, α)
    constraints.c!(state.x, b_ls.c)
    lagrangian(d, constraints.bounds, state.x_ls, b_ls.c, b_ls.bstate, state.μ)[2]
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
    bμ = _bv(x, bounds.iz, bounds.σz) +      # coords constrained by 0
         _bv(sx) +  # coords with other bounds
         _bv(sc)    # linear/nonlinear constr.
    μ*bμ
end
barrier_value(bounds::ConstraintBounds, x, bstate::BarrierStateVars, μ) =
    barrier_value(bounds, x, bstate.slack_x, bstate.slack_c, μ)
barrier_value(bounds::ConstraintBounds, state) =
    barrier_value(bounds, state.x, state.bstate.slack_x, state.bstate.slack_c, state.μ)
barrier_value(constraints::AbstractConstraintsFunction, state) =
    barrier_value(constraints.bounds, state)

# don't call this barrier_value because it lacks μ
function _bv(v, idx, σ)
    ret = loginf(one(eltype(σ))*one(eltype(v)))
    for (i,iv) in enumerate(idx)
        ret += loginf(σ[i]*v[iv])
    end
    -ret
end

_bv(v) = isempty(v) ? loginf(one(eltype(v))) : -sum(loginf, v)

loginf(δ) = δ > 0 ? log(δ) : -oftype(δ, Inf)

"""
    barrier_grad!(gx, bgrad, bounds, x, bstate, μ)
    barrier_grad!(gx, gsx, gsc, bounds, x, sx, sc, μ)

Compute the gradient of the barrier penalty at (`x`,`sx`,`sc`), where
`x` is the current position, `sx` are the coordinate slack variables,
and `sc` are the linear/nonlinear slack
variables. `bounds::ConstraintBounds` holds the parsed bounds.

The result is *added* to `gx`, `gsx`, and `gsc`, so these vectors
need to be initialized appropriately.
"""
function barrier_grad!(gx, gsx, gsc, bounds::ConstraintBounds, x, sx, sc, μ)
    barrier_grad!(view(gx, bounds.iz), view(x, bounds.iz), μ)
    barrier_grad!(gsx, sx, μ)
    barrier_grad!(gsc, sc, μ)
    nothing
end
barrier_grad!(gx, bgrad, bounds::ConstraintBounds, x, bstate, μ) =
    barrier_grad!(gx, bgrad.slack_x, bgrad.slack_c, bounds, x, bstate.slack_x, bstate.slack_c, μ)

function barrier_grad!(out, v, μ)
    for i = 1:length(out)
        out[i] -= μ/v[i]
    end
    nothing
end


## Computation of Lagrangian terms: equality constraints penalty

"""
    equality_violation([f=identity], bounds, x, c, bstate) -> val
    equality_violation([f=identity], bounds, x, c, sx, sc, λxE, λx, λc, λcE) -> val

Compute the sum of `f(v_i)`, where `v_i = λ_i*(target - observed)`
measures the difference between the current state and the
equality-constrained state. `bounds::ConstraintBounds` holds the
parsed bounds. `x` is the current position, `sx` are the coordinate
slack variables, and `sc` are the linear/nonlinear slack
variables. `c` holds the values of the linear-nonlinear constraints,
and the λ arguments hold the Lagrange multipliers for `x`, `sx`, `sc`, and
`c` respectively.
"""
function equality_violation(f, bounds::ConstraintBounds, x, c, sx, sc, λxE, λx, λc, λcE)
    ev = equality_violation(f, x, bounds.valx, bounds.eqx, λxE) +
         equality_violation(f, sx, x, bounds.ineqx, bounds.σx, bounds.bx, λx) +
         equality_violation(f, sc, c, bounds.ineqc, bounds.σc, bounds.bc, λc) +
         equality_violation(f, c, bounds.valc, bounds.eqc, λcE)
end
equality_violation(bounds::ConstraintBounds, x, c, sx, sc, λxE, λx, λc, λcE) =
    equality_violation(identity, bounds, x, c, sx, sc, λxE, λx, λc, λcE)
function equality_violation(f, bounds::ConstraintBounds, x, c, bstate::BarrierStateVars)
    equality_violation(f, bounds, x, c,
                       bstate.slack_x, bstate.slack_c, bstate.λxE, bstate.λx, bstate.λc, bstate.λcE)
end
equality_violation(bounds::ConstraintBounds, x, c, bstate::BarrierStateVars) =
    equality_violation(identity, bounds, x, c, bstate)
equality_violation(f, bounds::ConstraintBounds, state::AbstractBarrierState) =
    equality_violation(f, bounds, state.x, state.constr_c, state.bstate)
equality_violation(bounds::ConstraintBounds, state::AbstractBarrierState) =
    equality_violation(identity, bounds, state)
equality_violation(f, constraints::AbstractConstraintsFunction, state::AbstractBarrierState) =
    equality_violation(f, constraints.bounds, state)
equality_violation(constraints::AbstractConstraintsFunction, state::AbstractBarrierState) =
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
function equality_grad!(gx, gsx, gsc, gλxE, gλx, gλc, gλcE, bounds::ConstraintBounds, x, c, J, sx, sc, λxE, λx, λc, λcE)
    gx[bounds.eqx] = gx[bounds.eqx] - λxE
    equality_grad_var!(gsx, gx, bounds.ineqx, bounds.σx, λx)
    equality_grad_var!(gsc, gx, bounds.ineqc, bounds.σc, λc, J)
    equality_grad_var!(gx, bounds.eqc, λcE, J)
    equality_grad_λ!(gλxE, x, bounds.valx, bounds.eqx)
    equality_grad_λ!(gλx, sx, x, bounds.ineqx, bounds.σx, bounds.bx)
    equality_grad_λ!(gλc, sc, c, bounds.ineqc, bounds.σc, bounds.bc)
    equality_grad_λ!(gλcE, c, bounds.valc, bounds.eqc)
end
equality_grad!(gx, gb::BarrierStateVars, bounds::ConstraintBounds, x, c, J, b::BarrierStateVars) =
    equality_grad!(gx, gb.slack_x, gb.slack_c, gb.λxE, gb.λx, gb.λc, gb.λcE,
                   bounds, x, c, J,
                   b.slack_x, b.slack_c, b.λxE, b.λx, b.λc, b.λcE)

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
    gs[:] = gs + λ
    if !isempty(ineq)
        gx[:] = gx - view5(J, ineq, :)'*(λ.*σ)
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
        gx[:] = gx - view5(J, idx, :)'*λ
    end
    nothing
end

function equality_grad_λ!(gλ, v, target, idx)
    for (i,iv) in enumerate(idx)
        gλ[i] += target[i] - v[iv]
    end
    nothing
end

## Utilities for representing total state as single vector
function pack_vec(x, b::BarrierStateVars)
    n = length(x)
    for fn in fieldnames(b)
        n += length(getfield(b, fn))
    end
    vec = Array{eltype(x)}(n)
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
