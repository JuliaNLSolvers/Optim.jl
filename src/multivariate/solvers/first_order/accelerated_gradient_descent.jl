# http://stronglyconvex.com/blog/accelerated-gradient-descent.html
# TODO: Need to specify alphamax on each iteration
# Flip notation relative to Duckworth
# Start with x_{0}
# y_{t} = x_{t - 1} - alpha g(x_{t - 1})
# If converged, return y_{t}
# x_{t} = y_{t} + (t - 1.0) / (t + 2.0) * (y_{t} - y_{t - 1})

# L should be function or any other callable
struct AcceleratedGradientDescent{L} <: Optimizer
    linesearch!::L
    manifold::Manifold
end

Base.summary(::AcceleratedGradientDescent) = "Accelerated Gradient Descent"

#= uncomment for v0.8.0
AcceleratedGradientDescent(; linesearch = LineSearches.HagerZhang()) =
  AcceleratedGradientDescent(linesearch)
=#
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

function initial_state{T}(method::AcceleratedGradientDescent, options, d, initial_x::Array{T})
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)
    value_gradient!(d, initial_x)
    project_tangent!(method.manifold, gradient(d), initial_x)

    AcceleratedGradientDescentState(initial_x, # Maintain current state in state.x
                         copy(initial_x), # Maintain previous state in state.x_previous
                         T(NaN), # Store previous f in state.f_x_previous
                         0, # Iteration
                         copy(initial_x), # Maintain intermediary current state in state.y
                         similar(initial_x), # Maintain intermediary state in state.y_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::AcceleratedGradientDescentState{T}, method::AcceleratedGradientDescent)
    n = length(state.x)
    state.iteration += 1
    project_tangent!(method.manifold, gradient(d), state.x)
    # Search direction is always the negative gradient
    @simd for i in 1:n
        @inbounds state.s[i] = -gradient(d, i)
    end

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Record previous state
    copy!(state.x_previous, state.x)

    # Make one move in the direction of the gradient
    copy!(state.y_previous, state.y)
    @simd for i in 1:n
        @inbounds state.y[i] = state.x[i] + state.alpha * state.s[i]
    end
    retract!(method.manifold, state.y)

    # Update current position with Nesterov correction
    scaling = (state.iteration - 1) / (state.iteration + 2)
    @simd for i in 1:n
        @inbounds state.x[i] = state.y[i] + scaling * (state.y[i] - state.y_previous[i])
    end
    retract!(method.manifold, state.x)

    lssuccess == false # break on linesearch error
end
