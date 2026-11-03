struct Newton{IL,L} <: SecondOrderOptimizer
    alphaguess!::IL
    linesearch!::L
end

"""
# Newton
## Constructor
```julia
Newton(; alphaguess = LineSearches.InitialStatic(),
linesearch = LineSearches.HagerZhang())
```

## Description
The `Newton` method implements Newton's method for optimizing a function. To
ensure that each search direction is a direction of descent, an indefinite
Hessian is shifted by a multiple of the identity: twice the smallest one, in a
geometric sequence, for which its Cholesky factorization succeeds (cf. Algorithm
3.3 in Nocedal and Wright, 2006). Doubling it gives the most negative curvature
the opposite sign rather than nearly zero. See Nocedal and Wright (ch. 6, 1999)
for a discussion of Newton's method in practice.

The Hessian is used only through Cholesky factorizations, so besides dense
matrices it can be a sparse or a user-defined structured matrix.

## References
 - Nocedal, J. and S. J. Wright (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
function Newton(;
    alphaguess = LineSearches.InitialStatic(), # Good default for Newton
    linesearch = LineSearches.HagerZhang(),
)    # Good default for Newton
    Newton(_alphaguess(alphaguess), linesearch)
end

Base.summary(io::IO, ::Newton) = print(io, "Newton's Method")

mutable struct NewtonState{Tx,Tg,TH,T} <: AbstractOptimizerState
    x::Tx
    g_x::Tg
    H_x::TH
    f_x::T
    x_previous::Tx
    f_x_previous::T
    s::Tx
    # Trial iterate produced by update_state! / update_fgh!. Committed to
    # state.x / state.g_x / state.H_x / state.f_x by accept_step! once validated.
    x_candidate::Tx
    g_candidate::Tg
    H_candidate::TH
    f_candidate::T
    # Workspace for factorizing (shifted copies of) H_x in place
    const W::TH
    @add_linesearch_fields()
end

function initial_state(method::Newton, options, d, x0)
    f_x, g_x, H_x = NLSolversBase.value_gradient_hessian!(d, x0)

    NewtonState(
        copy(x0), # Maintain current state in state.x
        copy(g_x), # Maintain current gradient in state.g_x
        copy(H_x), # Maintain current Hessian in state.H_x
        f_x, # Maintain current f in state.f_x
        fill!(similar(x0), NaN), # Maintain previous state in state.x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        fill!(similar(x0), NaN), # Maintain current search direction in state.s
        fill!(similar(x0), NaN), # Trial iterate in state.x_candidate
        fill!(similar(g_x), NaN), # Trial gradient in state.g_candidate
        similar(H_x), # Trial Hessian in state.H_candidate
        oftype(f_x, NaN), # Trial f value in state.f_candidate
        similar(H_x), # Factorization workspace in state.W
        @initial_linesearch()...,
    )
end

function update_state!(d, state::NewtonState, method::Newton)
    # The search direction solves (H + τI)s = -g with H + τI positive definite
    # (see `_modified_cholesky!`), so it is a direction of descent. `vec` lets x
    # be an array of any shape.
    F = _modified_cholesky!(state.W, state.H_x)
    if F === nothing
        # Only a non-finite H defeats every shift; fall back to steepest descent
        state.s .= .-state.g_x
    else
        _ldiv!(vec(state.s), F, vec(state.g_x))
        state.s .= .-state.s
    end
    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Propose trial iterate (do NOT mutate state.x; accept_step! commits)
    copyto!(state.x_candidate, state.x_ls)
    return !lssuccess # break on linesearch error
end

function update_fgh!(d, state::NewtonState, method::Newton)
    f_c, g_c, H_c = NLSolversBase.value_gradient_hessian!(d, state.x_candidate)
    state.f_candidate = f_c
    copyto!(state.g_candidate, g_c)
    copyto!(state.H_candidate, H_c)
    return nothing
end

function accept_step!(d, state::NewtonState, method::Newton, options)
    if !isfinite(state.f_candidate) ||
       !all(isfinite, state.g_candidate) ||
       !all(isfinite, state.H_candidate) ||
       !all(isfinite, state.x_candidate)
        return false
    end
    copyto!(state.x, state.x_candidate)
    copyto!(state.g_x, state.g_candidate)
    copyto!(state.H_x, state.H_candidate)
    state.f_x = state.f_candidate
    return true
end

function trace!(tr, d, state::NewtonState, iteration::Integer, ::Newton, options::Options, curr_time = time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g_x)
        dt["h(x)"] = copy(state.H_x)
        dt["Current step size"] = state.alpha
    end
    update!(
        tr,
        iteration,
        state.f_x,
        g_residual(state),
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
    )
end
