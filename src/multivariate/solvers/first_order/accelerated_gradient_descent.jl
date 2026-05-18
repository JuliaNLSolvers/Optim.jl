# http://stronglyconvex.com/blog/accelerated-gradient-descent.html
# TODO: Need to specify alphamax on each iteration
# Flip notation relative to Duckworth
# Start with x_{0}
# y_{t} = x_{t - 1} - alpha g(x_{t - 1})
# If converged, return y_{t}
# x_{t} = y_{t} + (t - 1.0) / (t + 2.0) * (y_{t} - y_{t - 1})

struct AcceleratedGradientDescent{IL,L} <: FirstOrderOptimizer
    alphaguess!::IL
    linesearch!::L
    manifold::Manifold
end

Base.summary(io::IO, ::AcceleratedGradientDescent) = print(io, "Accelerated Gradient Descent")

function AcceleratedGradientDescent(;
    alphaguess = LineSearches.InitialPrevious(), # TODO: investigate good defaults
    linesearch = LineSearches.HagerZhang(),        # TODO: investigate good defaults
    manifold::Manifold = Flat(),
)
    AcceleratedGradientDescent(_alphaguess(alphaguess), linesearch, manifold)
end

mutable struct AcceleratedGradientDescentState{T,Tx,Tg} <: AbstractOptimizerState
    x::Tx
    g_x::Tg
    f_x::T
    x_previous::Tx
    f_x_previous::T
    iteration::Int
    y::Tx
    y_previous::Tx
    s::Tx
    # Trial iterates produced by update_state! / update_fgh!. Committed to
    # state.x / state.g_x / state.f_x / state.y by accept_step! once validated.
    x_candidate::Tx
    y_candidate::Tx
    g_candidate::Tg
    f_candidate::T
    @add_linesearch_fields()
end

function initial_state(
    method::AcceleratedGradientDescent,
    ::Options,
    d,
    x0::AbstractArray,
)
    x0 = copy(x0)
    retract!(method.manifold, x0)
    f_x, g_x = NLSolversBase.value_gradient!(d, x0)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, x0)

    AcceleratedGradientDescentState(
        x0, # Maintain current state in state.x
        g_x, # Maintain current gradient in state.g_x
        f_x, # Maintain current f in state.f_x
        fill!(similar(x0), NaN), # Maintain previous state in state.x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        0, # Iteration
        copy(x0), # Maintain intermediary current state in state.y
        fill!(similar(x0), NaN), # Maintain intermediary state in state.y_previous
        fill!(similar(x0), NaN), # Maintain current search direction in state.s
        fill!(similar(x0), NaN), # Trial iterate in state.x_candidate
        fill!(similar(x0), NaN), # Trial y iterate in state.y_candidate
        fill!(similar(g_x), NaN), # Trial gradient in state.g_candidate
        oftype(f_x, NaN), # Trial f value in state.f_candidate
        @initial_linesearch()...,
    )
end

function update_state!(
    d,
    state::AcceleratedGradientDescentState,
    method::AcceleratedGradientDescent,
)
    # Search direction is always the negative gradient
    state.s .= .-state.g_x

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Propose trial intermediary y (do NOT mutate state.y; accept_step! commits)
    state.y_candidate .= state.x .+ state.alpha .* state.s
    retract!(method.manifold, state.y_candidate)

    # Propose trial position with Nesterov correction. iteration is incremented
    # on accept so the scaling here uses the would-be next iteration index.
    next_iteration = state.iteration + 1
    scaling = (next_iteration - 1) / (next_iteration + 2)
    state.x_candidate .= state.y_candidate .+ scaling .* (state.y_candidate .- state.y)
    retract!(method.manifold, state.x_candidate)

    return !lssuccess # break on linesearch error
end

function update_fgh!(
    d,
    state::AcceleratedGradientDescentState,
    method::AcceleratedGradientDescent,
)
    f_c, g_c = NLSolversBase.value_gradient!(d, state.x_candidate)
    copyto!(state.g_candidate, g_c)
    project_tangent!(method.manifold, state.g_candidate, state.x_candidate)
    state.f_candidate = f_c
    return nothing
end

function accept_step!(
    d,
    state::AcceleratedGradientDescentState,
    method::AcceleratedGradientDescent,
    options,
)
    if !isfinite(state.f_candidate) ||
       !all(isfinite, state.g_candidate) ||
       !all(isfinite, state.x_candidate) ||
       !all(isfinite, state.y_candidate)
        return false
    end
    copyto!(state.y_previous, state.y)
    copyto!(state.y, state.y_candidate)
    copyto!(state.x, state.x_candidate)
    copyto!(state.g_x, state.g_candidate)
    state.f_x = state.f_candidate
    state.iteration += 1
    return true
end

function trace!(
    tr,
    d,
    state::AcceleratedGradientDescentState,
    iteration::Integer,
    method::AcceleratedGradientDescent,
    options::Options,
    curr_time = time(),
)
    common_trace!(tr, d, state, iteration, method, options, curr_time)
end

function default_options(method::AcceleratedGradientDescent)
    (; allow_f_increases = true)
end
