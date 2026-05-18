# See p. 280 of Murphy's Machine Learning
# x_k1 = x_k - alpha * gr + mu * (x - x_previous)

struct MomentumGradientDescent{Tf,IL,L} <: FirstOrderOptimizer
    mu::Tf
    alphaguess!::IL
    linesearch!::L
    manifold::Manifold
end

Base.summary(io::IO, ::MomentumGradientDescent) = print(io, "Momentum Gradient Descent")

function MomentumGradientDescent(;
    mu::Real = 0.01,
    alphaguess = LineSearches.InitialPrevious(), # TODO: investigate good defaults
    linesearch = LineSearches.HagerZhang(),        # TODO: investigate good defaults
    manifold::Manifold = Flat(),
)
    MomentumGradientDescent(mu, _alphaguess(alphaguess), linesearch, manifold)
end

mutable struct MomentumGradientDescentState{Tx,Tg,T} <: AbstractOptimizerState
    x::Tx
    g_x::Tg
    f_x::T
    x_previous::Tx
    x_momentum::Tx
    f_x_previous::T
    s::Tx
    # Trial iterate produced by update_state! / update_fgh!. Committed to
    # state.x / state.g_x / state.f_x by accept_step! once validated.
    x_candidate::Tx
    g_candidate::Tg
    f_candidate::T
    @add_linesearch_fields()
end

function initial_state(method::MomentumGradientDescent, ::Options, d, x0::AbstractArray)
    # Compute function value and gradient
    x0 = copy(x0)
    retract!(method.manifold, x0)
    f_x, g_x = value_gradient!(d, x0)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, x0)

    MomentumGradientDescentState(
        x0, # Maintain current state in state.x
        g_x, # Maintain current gradient in state.g_x
        f_x, # Maintain current f in state.f_x
        copy(x0), # Maintain previous state in state.x_previous
        fill!(similar(x0), NaN), # Record momentum correction direction in state.x_momentum
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        fill!(similar(x0), NaN), # Maintain current search direction in state.s
        fill!(similar(x0), NaN), # Trial iterate in state.x_candidate
        fill!(similar(g_x), NaN), # Trial gradient in state.g_candidate
        oftype(f_x, NaN), # Trial f value in state.f_candidate
        @initial_linesearch()...,
    )
end

function update_state!(
    d,
    state::MomentumGradientDescentState,
    method::MomentumGradientDescent,
)
    # Search direction is always the negative gradient
    state.s .= .-state.g_x

    # Update momentum (uses committed state.x and state.x_previous)
    state.x_momentum .= state.x .- state.x_previous

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Propose trial iterate (do NOT mutate state.x; accept_step! commits)
    state.x_candidate .= state.x .+ state.alpha .* state.s .+ method.mu .* state.x_momentum
    retract!(method.manifold, state.x_candidate)

    return !lssuccess # break on linesearch error
end

function update_fgh!(
    d,
    state::MomentumGradientDescentState,
    method::MomentumGradientDescent,
)
    f_c, g_c = NLSolversBase.value_gradient!(d, state.x_candidate)
    copyto!(state.g_candidate, g_c)
    project_tangent!(method.manifold, state.g_candidate, state.x_candidate)
    state.f_candidate = f_c
    return nothing
end

function accept_step!(
    d,
    state::MomentumGradientDescentState,
    method::MomentumGradientDescent,
    options,
)
    if !isfinite(state.f_candidate) ||
       !all(isfinite, state.g_candidate) ||
       !all(isfinite, state.x_candidate)
        return false
    end
    copyto!(state.x, state.x_candidate)
    copyto!(state.g_x, state.g_candidate)
    state.f_x = state.f_candidate
    return true
end

function trace!(
    tr,
    d,
    state::MomentumGradientDescentState,
    iteration::Integer,
    method::MomentumGradientDescent,
    options::Options,
    curr_time = time(),
)
    common_trace!(tr, d, state, iteration, method, options, curr_time)
end

function default_options(method::MomentumGradientDescent)
    (; allow_f_increases = true)
end
