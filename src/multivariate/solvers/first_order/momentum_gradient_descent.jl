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
    @add_linesearch_fields()
end

function initial_state(method::MomentumGradientDescent, ::Options, d, initial_x::AbstractArray)
    # Compute function value and gradient
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)
    f_x, g_x = value_gradient!(d, initial_x)
    project_tangent!(method.manifold, g_x, initial_x)

    MomentumGradientDescentState(
        initial_x, # Maintain current state in state.x
        copy(g_x), # Maintain current gradient in state.g_x
        f_x, # Maintain current f in state.f_x 
        copy(initial_x), # Maintain previous state in state.x_previous
        fill!(similar(initial_x), NaN), # Record momentum correction direction in state.x_momentum
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        fill!(similar(initial_x), NaN), # Maintain current search direction in state.s
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

    # Update momentum
    state.x_momentum .= state.x .- state.x_previous

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Update state
    state.x .+= state.alpha .* state.s .+ method.mu .* state.x_momentum
    retract!(method.manifold, state.x)

    return !lssuccess # break on linesearch error
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
