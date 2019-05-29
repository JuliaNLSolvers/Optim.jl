# See p. 280 of Murphy's Machine Learning
# x_k1 = x_k - alpha * gr + mu * (x - x_previous)

struct MomentumGradientDescent{Tf, IL,L} <: FirstOrderOptimizer
    mu::Tf
    alphaguess!::IL
    linesearch!::L
    manifold::Manifold
end

Base.summary(::MomentumGradientDescent) = "Momentum Gradient Descent"

function MomentumGradientDescent(; mu::Real = 0.01,
                                 alphaguess = LineSearches.InitialPrevious(), # TODO: investigate good defaults
                                 linesearch = LineSearches.HagerZhang(),        # TODO: investigate good defaults
                                 manifold::Manifold=Flat())
    MomentumGradientDescent(mu, alphaguess, linesearch, manifold)
end

mutable struct MomentumGradientDescentState{Tx, T} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    x_momentum::Tx
    f_x_previous::T
    s::Tx
    @add_linesearch_fields()
end

function initial_state(method::MomentumGradientDescent, options, d, initial_x)
    T = eltype(initial_x)
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)

    value_gradient!!(d, initial_x)

    project_tangent!(method.manifold, gradient(d), initial_x)

    MomentumGradientDescentState(initial_x, # Maintain current state in state.x
                                 copy(initial_x), # Maintain previous state in state.x_previous
                                 similar(initial_x), # Record momentum correction direction in state.x_momentum
                                 real(T)(NaN), # Store previous f in state.f_x_previous
                                 similar(initial_x), # Maintain current search direction in state.s
                                 @initial_linesearch()...)
end

function update_state!(d, state::MomentumGradientDescentState, method::MomentumGradientDescent)
    project_tangent!(method.manifold, gradient(d), state.x)
    # Search direction is always the negative gradient
    state.s .= .-gradient(d)

    # Update position, and backup current one
    state.x_momentum .= state.x .- state.x_previous

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    state.x .+= state.alpha.*state.s .+ method.mu.*state.x_momentum
    retract!(method.manifold, state.x)
    lssuccess == false # break on linesearch error
end

function trace!(tr, d, state, iteration, method::MomentumGradientDescent, options, curr_time=time())
  common_trace!(tr, d, state, iteration, method, options, curr_time)
end

function default_options(method::MomentumGradientDescent)
    Dict(:allow_f_increases => true)
end
