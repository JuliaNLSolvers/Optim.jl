# http://stronglyconvex.com/blog/accelerated-gradient-descent.html
# TODO: Need to specify alphamax on each iteration
# Flip notation relative to Duckworth
# Start with x_{0}
# y_{t} = x_{t - 1} - alpha g(x_{t - 1})
# If converged, return y_{t}
# x_{t} = y_{t} + (t - 1.0) / (t + 2.0) * (y_{t} - y_{t - 1})

struct AcceleratedGradientDescent{IL, L} <: FirstOrderOptimizer
    alphaguess!::IL
    linesearch!::L
    manifold::Manifold
end

Base.summary(::AcceleratedGradientDescent) = "Accelerated Gradient Descent"

function AcceleratedGradientDescent(;
                                    alphaguess = LineSearches.InitialPrevious(), # TODO: investigate good defaults
                                    linesearch = LineSearches.HagerZhang(),        # TODO: investigate good defaults
                                    manifold::Manifold=Flat())
    AcceleratedGradientDescent(alphaguess, linesearch, manifold)
end

mutable struct AcceleratedGradientDescentState{T, Tx} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    f_x_previous::T
    iteration::Int
    y::Tx
    y_previous::Tx
    s::Tx
    @add_linesearch_fields()
end

function initial_state(method::AcceleratedGradientDescent, options, d, initial_x::AbstractArray{T}) where T
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)

    value_gradient!!(d, initial_x)

    project_tangent!(method.manifold, gradient(d), initial_x)

    AcceleratedGradientDescentState(copy(initial_x), # Maintain current state in state.x
                         copy(initial_x), # Maintain previous state in state.x_previous
                         real(T)(NaN), # Store previous f in state.f_x_previous
                         0, # Iteration
                         copy(initial_x), # Maintain intermediary current state in state.y
                         similar(initial_x), # Maintain intermediary state in state.y_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         @initial_linesearch()...)
end

function update_state!(d, state::AcceleratedGradientDescentState, method::AcceleratedGradientDescent)
    value_gradient!(d, state.x)
    state.iteration += 1
    project_tangent!(method.manifold, gradient(d), state.x)
    # Search direction is always the negative gradient
    state.s .= .-gradient(d)

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Make one move in the direction of the gradient
    copyto!(state.y_previous, state.y)
    state.y .= state.x .+ state.alpha.*state.s
    retract!(method.manifold, state.y)

    # Update current position with Nesterov correction
    scaling = (state.iteration - 1) / (state.iteration + 2)
    state.x .= state.y .+ scaling.*(state.y .- state.y_previous)
    retract!(method.manifold, state.x)

    lssuccess == false # break on linesearch error
end

function trace!(tr, d, state, iteration, method::AcceleratedGradientDescent, options, curr_time=time())
  common_trace!(tr, d, state, iteration, method, options, curr_time)
end

function default_options(method::AcceleratedGradientDescent)
    Dict(:allow_f_increases => true)
end
