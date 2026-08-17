"""
    Newton(; alphaguess=LineSearches.InitialStatic(),
        linesearch=LineSearches.HagerZhang())

Newton's method uses the objective Hessian to compute a second-order search
direction and a line search to select the step length. A factorization from
`PositiveFactorizations.jl` modifies indefinite Hessians so that the search
direction remains a descent direction.

# Keyword Arguments

- `alphaguess`: Initial step-length guess used by `linesearch`.
- `linesearch`: Line-search method used to choose the step length.

# Fields

- `alphaguess!`: Normalized initial step-length guess callable.
- `linesearch!`: Line-search callable.

# Returns

An initialized `Newton` optimizer that can be passed to [`optimize`](@ref).
The objective supplied to `optimize` must provide a Hessian, either directly
or through the selected automatic-differentiation configuration.

# Examples

```julia
using Optim
using LinearAlgebra: I

f(x) = sum(abs2, x)
g!(G, x) = (G .= 2 .* x)
h!(H, x) = (H .= 2 .* I(length(x)))
result = optimize(f, g!, h!, [1.0, -1.0], Newton())
Optim.minimizer(result)
```

# References

- Nocedal, J. and Wright, S. J. (1999), *Numerical Optimization*.
"""
struct Newton{IL,L} <: SecondOrderOptimizer
    alphaguess!::IL
    linesearch!::L
end

function Newton(;
    alphaguess = LineSearches.InitialStatic(), # Good default for Newton
    linesearch = LineSearches.HagerZhang(),
)    # Good default for Newton
    Newton(_alphaguess(alphaguess), linesearch)
end

Base.summary(io::IO, ::Newton) = print(io, "Newton's Method")

mutable struct NewtonState{Tx,Tg,TH,T,F<:Cholesky} <: AbstractOptimizerState
    x::Tx
    g_x::Tg
    H_x::TH
    f_x::T
    x_previous::Tx
    f_x_previous::T
    F::F
    s::Tx
    # Trial iterate produced by update_state! / update_fgh!. Committed to
    # state.x / state.g_x / state.H_x / state.f_x by accept_step! once validated.
    x_candidate::Tx
    g_candidate::Tg
    H_candidate::TH
    f_candidate::T
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
        Cholesky(similar(H_x, 0, 0), :U, 0),
        fill!(similar(x0), NaN), # Maintain current search direction in state.s
        fill!(similar(x0), NaN), # Trial iterate in state.x_candidate
        fill!(similar(g_x), NaN), # Trial gradient in state.g_candidate
        similar(H_x), # Trial Hessian in state.H_candidate
        oftype(f_x, NaN), # Trial f value in state.f_candidate
        @initial_linesearch()...,
    )
end

function update_state!(d, state::NewtonState, method::Newton)
    # Search direction is always the negative gradient divided by
    # a matrix encoding the absolute values of the curvatures
    # represented by H. It deviates from the usual "add a scaled
    # identity matrix" version of the modified Newton method. More
    # information can be found in the discussion at issue #153.

    if state.H_x isa AbstractSparseMatrix
        state.s .= .-(state.H_x \ convert(Vector, state.g_x))
    else
        state.F = cholesky!(Positive, state.H_x)
        if state.g_x isa Array
            # is this actually StridedArray?
            ldiv!(state.s, state.F, state.g_x)
            state.s .= .-state.s
        else
            # not Array, we can't do inplace ldiv
            gv = Vector{eltype(state.g_x)}(undef, length(state.g_x))
            gv .= .-state.g_x
            copyto!(state.s, state.F \ gv)
        end
    end
    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Propose trial iterate (do NOT mutate state.x; accept_step! commits)
    @. state.x_candidate = state.x + state.alpha * state.s
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
