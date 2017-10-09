# http://stronglyconvex.com/blog/accelerated-gradient-descent.html
# TODO: Need to specify alphamax on each iteration
# Flip notation relative to Duckworth
# Start with x_{0}
# y_{t} = x_{t - 1} - alpha g(x_{t - 1})
# If converged, return y_{t}
# x_{t} = y_{t} + (t - 1.0) / (t + 2.0) * (y_{t} - y_{t - 1})

struct AcceleratedGradientDescent{L} <: Optimizer
    linesearch!::L
    manifold::Manifold
end

Base.summary(::AcceleratedGradientDescent) = "Accelerated Gradient Descent"

function AcceleratedGradientDescent(; linesearch = LineSearches.HagerZhang(), manifold::Manifold=Flat())
    AcceleratedGradientDescent(linesearch, manifold)
end

mutable struct AcceleratedGradientDescentState{T,N}
    x::Array{T,N}
    x_previous::Array{T,N}
    f_x_previous::T
    iteration::Int
    y::Array{T,N}
    y_previous::Array{T,N}
    s::Array{T,N}
    @add_linesearch_fields()
end

function initial_state(method::AcceleratedGradientDescent, options, d, initial_x::Array{T}) where T
    initial_x = copy(initial_x)
    retract!(method.manifold, real_to_complex(d,initial_x))
    value_gradient!(d, initial_x)
    project_tangent!(method.manifold, real_to_complex(d,gradient(d)), real_to_complex(d,initial_x))

    AcceleratedGradientDescentState(initial_x, # Maintain current state in state.x
                         copy(initial_x), # Maintain previous state in state.x_previous
                         T(NaN), # Store previous f in state.f_x_previous
                         0, # Iteration
                         copy(initial_x), # Maintain intermediary current state in state.y
                         similar(initial_x), # Maintain intermediary state in state.y_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!(d, state::AcceleratedGradientDescentState{T}, method::AcceleratedGradientDescent) where T
    state.iteration += 1
    project_tangent!(method.manifold, real_to_complex(d,gradient(d)), real_to_complex(d,state.x))
    # Search direction is always the negative gradient
    state.s .= .-gradient(d)

    # Record previous state
    copy!(state.x_previous, state.x)
    state.f_x_previous = value(d)

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Make one move in the direction of the gradient
    copy!(state.y_previous, state.y)
    state.y .= state.x .+ state.alpha.*state.s
    retract!(method.manifold, real_to_complex(d,state.y))

    # Update current position with Nesterov correction
    scaling = (state.iteration - 1) / (state.iteration + 2)
    state.x .= state.y .+ scaling.*(state.y .- state.y_previous)
    retract!(method.manifold, real_to_complex(d,state.x))

    lssuccess == false # break on linesearch error
end

function assess_convergence(state::AcceleratedGradientDescentState, d, options)
  default_convergence_assessment(state, d, options)
end


function trace!(tr, d, state, iteration, method::AcceleratedGradientDescent, options)
  common_trace!(tr, d, state, iteration, method, options)
end
