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
    initial_x::AbstractArray{T},
) where {T}
    # Compute function value and gradient
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)
    f_x, g_x = NLSolversBase.value_gradient!(d, initial_x)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, initial_x)

    GradientDescentState(
        initial_x, # Maintain current state in state.x
        g_x, # Maintain current gradient in state.g_x
        f_x, # Maintain current f in state.f_x
        fill!(similar(initial_x), NaN), # Maintain previous state in state.x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        fill!(similar(initial_x), NaN), # Maintain current search direction in state.s
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

    # Update current position # x = x + alpha * s
    @. state.x = state.x + state.alpha * state.s
    retract!(method.manifold, state.x)

    return !lssuccess # break on linesearch error
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
