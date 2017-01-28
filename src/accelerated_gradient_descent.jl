# http://stronglyconvex.com/blog/accelerated-gradient-descent.html
# TODO: Need to specify alphamax on each iteration
# Flip notation relative to Duckworth
# Start with x_{0}
# y_{t} = x_{t - 1} - alpha g(x_{t - 1})
# If converged, return y_{t}
# x_{t} = y_{t} + (t - 1.0) / (t + 2.0) * (y_{t} - y_{t - 1})

immutable AcceleratedGradientDescent{L<:Function} <: Optimizer
    linesearch!::L
end

#= uncomment for v0.8.0
AcceleratedGradientDescent(; linesearch = LineSearches.hagerzhang!) =
  AcceleratedGradientDescent(linesearch)
=#
function AcceleratedGradientDescent(; linesearch! = nothing,
                                      linesearch = LineSearches.hagerzhang!)
    linesearch = get_linesearch(linesearch!, linesearch)
    AcceleratedGradientDescent(linesearch)
end

type AcceleratedGradientDescentState{T,N,G}
    @add_generic_fields()
    x_previous::Array{T,N}
    g::G
    f_x_previous::T
    iteration::Int
    y::Array{T,N}
    y_previous::Array{T,N}
    s::Array{T,N}
    @add_linesearch_fields()
end

function initial_state{T}(method::AcceleratedGradientDescent, options, d, initial_x::Array{T})
    g = similar(initial_x)
    f_x = d.fg!(initial_x, g)

    AcceleratedGradientDescentState("Accelerated Gradient Descent",
                         length(initial_x),
                         copy(initial_x), # Maintain current state in state.x
                         f_x, # Store current f in state.f_x
                         1, # Track f calls in state.f_calls
                         1, # Track g calls in state.g_calls
                         0, # Track h calls in state.h_calls
                         copy(initial_x), # Maintain previous state in state.x_previous
                         g, # Store current gradient in state.g
                         T(NaN), # Store previous f in state.f_x_previous
                         0, # Iteration
                         similar(initial_x), # Maintain intermediary current state in state.y
                         similar(initial_x), # Maintain intermediary state in state.y_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::AcceleratedGradientDescentState{T}, method::AcceleratedGradientDescent)
    state.iteration += 1
    # Search direction is always the negative gradient
    @simd for i in 1:state.n
        @inbounds state.s[i] = -state.g[i]
    end

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Record previous state
    copy!(state.x_previous, state.x)

    # Make one move in the direction of the gradient
    copy!(state.y_previous, state.y)
    @simd for i in 1:state.n
        @inbounds state.y[i] = state.x[i] + state.alpha * state.s[i]
    end

    # Update current position with Nesterov correction
    scaling = (state.iteration - 1) / (state.iteration + 2)
    @simd for i in 1:state.n
        @inbounds state.x[i] = state.y[i] + scaling * (state.y[i] - state.y_previous[i])
    end

    # Update the function value and gradient
    state.f_x_previous, state.f_x = state.f_x, d.fg!(state.x, state.g)
    state.f_calls, state.g_calls = state.f_calls + 1, state.g_calls + 1

    lssuccess == false # break on linesearch error
end
