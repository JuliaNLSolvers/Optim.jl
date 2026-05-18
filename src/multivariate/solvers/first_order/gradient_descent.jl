struct GradientDescent{IL,L,T,Tprep} <: FirstOrderOptimizer
    alphaguess!::IL
    linesearch!::L
    P::T
    precondprep!::Tprep
    manifold::Manifold
end

Base.summary(io::IO, ::GradientDescent) = print(io, "Gradient Descent")

"""
# Gradient Descent
## Constructor
```julia
GradientDescent(; alphaguess = LineSearches.InitialHagerZhang(),
linesearch = LineSearches.HagerZhang(),
P = nothing,
precondprep = Returns(nothing),
manifold = Flat())
```
Keywords are used to control choice of line search, and preconditioning.

## Description
The `GradientDescent` method is a simple gradient descent algorithm, that is the
search direction is simply the negative gradient at the current iterate, and
then a line search step is used to compute the final step. See Nocedal and
Wright (ch. 2.2, 1999) for an explanation of the approach.

## References
 - Nocedal, J. and Wright, S. J. (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
function GradientDescent(;
    alphaguess = LineSearches.InitialPrevious(), # TODO: Investigate good defaults.
    linesearch = LineSearches.HagerZhang(),      # TODO: Investigate good defaults
    P = nothing,
    precondprep = Returns(nothing),
    manifold::Manifold = Flat(),
)
    GradientDescent(_alphaguess(alphaguess), linesearch, P, precondprep, manifold)
end

mutable struct GradientDescentState{Tx,Tg,T} <: AbstractOptimizerState
    x::Tx
    g_x::Tg
    f_x::T
    x_previous::Tx
    f_x_previous::T
    s::Tx
    # Trial iterate produced by update_state! / update_fgh!. Committed to
    # state.x / state.g_x / state.f_x by accept_step! once validated.
    x_candidate::Tx
    g_candidate::Tg
    f_candidate::T
    @add_linesearch_fields()
end

function reset!(method::GradientDescent, state::GradientDescentState, obj, x)
    # Update function value and gradient
    copyto!(state.x, x)
    retract!(method.manifold, state.x)
    f_x, g_x = NLSolversBase.value_gradient!(obj, state.x)
    copyto!(state.g_x, g_x)
    project_tangent!(method.manifold, state.g_x, state.x)
    state.f_x = f_x

    # Delete history
    fill!(state.x_previous, NaN)
    state.f_x_previous = oftype(state.f_x_previous, NaN)
    fill!(state.s, NaN)

    return nothing
end
function initial_state(
    method::GradientDescent,
    ::Options,
    d,
    x0::AbstractArray{T},
) where {T}
    # Compute function value and gradient
    x0 = copy(x0)
    retract!(method.manifold, x0)
    f_x, g_x = NLSolversBase.value_gradient!(d, x0)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, x0)

    GradientDescentState(
        x0, # Maintain current state in state.x
        g_x, # Maintain current gradient in state.g_x
        f_x, # Maintain current f in state.f_x
        fill!(similar(x0), NaN), # Maintain previous state in state.x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        fill!(similar(x0), NaN), # Maintain current search direction in state.s
        fill!(similar(x0), NaN), # Trial iterate in state.x_candidate
        fill!(similar(g_x), NaN), # Trial gradient in state.g_candidate
        oftype(f_x, NaN), # Trial f value in state.f_candidate
        @initial_linesearch()...,
    )
end

function update_state!(d, state::GradientDescentState{T}, method::GradientDescent) where {T}
    # Search direction is always the negative preconditioned gradient
    _precondition!(state.s, method, state.x, state.g_x)
    rmul!(state.s, eltype(state.s)(-1))
    if method.P !== nothing
        project_tangent!(method.manifold, state.s, state.x)
    end

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Propose trial iterate (do NOT mutate state.x; accept_step! commits)
    @. state.x_candidate = state.x + state.alpha * state.s
    retract!(method.manifold, state.x_candidate)

    return !lssuccess # break on linesearch error
end

function update_fgh!(d, state::GradientDescentState, method::GradientDescent)
    f_c, g_c = NLSolversBase.value_gradient!(d, state.x_candidate)
    copyto!(state.g_candidate, g_c)
    project_tangent!(method.manifold, state.g_candidate, state.x_candidate)
    state.f_candidate = f_c
    return nothing
end

function accept_step!(d, state::GradientDescentState, method::GradientDescent, options)
    if !isfinite(state.f_candidate) ||
       !all(isfinite, state.g_candidate) ||
       !all(isfinite, state.x_candidate)
        return false
    end
    # state.x_previous / state.f_x_previous were captured by perform_linesearch!
    # before the step was proposed, so they already hold the prior accepted values.
    copyto!(state.x, state.x_candidate)
    copyto!(state.g_x, state.g_candidate)
    state.f_x = state.f_candidate
    return true
end

function trace!(
    tr,
    d,
    state::GradientDescentState,
    iteration::Integer,
    method::GradientDescent,
    options::Options,
    curr_time = time(),
)
    common_trace!(tr, d, state, iteration, method, options, curr_time)
end
