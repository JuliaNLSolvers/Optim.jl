# http://stronglyconvex.com/blog/accelerated-gradient-descent.html
# TODO: Need to specify alphamax on each iteration
# Flip notation relative to Duckworth
# Start with x_{0}
# y_{t} = x_{t - 1} - alpha g(x_{t - 1})
# If converged, return y_{t}
# x_{t} = y_{t} + (t - 1.0) / (t + 2.0) * (y_{t} - y_{t - 1})


immutable AcceleratedGradientDescent <: Optimizer
    linesearch!::Function
end

AcceleratedGradientDescent(; linesearch!::Function = hz_linesearch!) =
  AcceleratedGradientDescent(linesearch!)

  method_string(method::AcceleratedGradientDescent) = "Accelerated Gradient Descent"

  type AcceleratedGradientDescentState{T}
      n::Int64
      x::Array{T}
      x_previous::Array{T}
      y::Array{T}
      y_previous::Array{T}
      g::Array{T}
      f_x::T
      f_x_previous::T
      s::Array{T}
      x_ls::Array{T}
      g_ls::Array{T}
      alpha::T
      mayterminate::Bool
      f_calls::Int64
      g_calls::Int64
      lsr
  end

function initialize_state{T}(method::AcceleratedGradientDescent, options, d, initial_x::Array{T})
    g = similar(initial_x)
    f_x = d.fg!(initial_x, g)

    AcceleratedGradientDescentState(length(initial_x),
                         copy(initial_x), # Maintain current state in state.x
                         copy(initial_x), # Maintain current state in state.x_previous
                         copy(initial_x), # Maintain intermediary current state in state.y
                         copy(initial_x), # Maintain intermediary state in state.y_previous
                         g, # Store current gradient in state.g
                         f_x, # Store current f in state.f_x
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         similar(initial_x), # Buffer of x for line search in state.x_ls
                         similar(initial_x), # Buffer of g for line search in state.g_ls
                         alphainit(one(T), initial_x, g, f_x), # Keep track of step size in state.alpha
                         false, # state.mayterminate
                         1, # Track f calls in state.f_calls
                         1, # Track g calls in state.g_calls
                         LineSearchResults(T)) # Maintain a cache for line search results in state.lsr
end

function update!{T}(d, state::AcceleratedGradientDescentState{T}, method::AcceleratedGradientDescent)

    # Search direction is always the negative gradient
    @simd for i in 1:state.n
        @inbounds state.s[i] = -state.g[i]
    end

    # Refresh the line search cache
    dphi0 = vecdot(state.g, state.s)
    clear!(state.lsr)
    push!(state.lsr, zero(T), state.f_x, dphi0)

    # Determine the distance of movement along the search line
    state.alpha, f_update, g_update =
      method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls, state.lsr, state.alpha, state.mayterminate)
    state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update

    # Make one move in the direction of the gradient
    copy!(state.y_previous, state.y)
    @simd for i in 1:state.n
        @inbounds state.y[i] = state.x_previous[i] + state.alpha * state.s[i]
    end

    # Record previous state
    copy!(state.x_previous, state.x)

    # Update current position with Nesterov correction
    scaling = (iteration - 1) / (iteration + 2)
    @simd for i in 1:n
        @inbounds state.x[i] = state.y[i] + scaling * (state.y[i] - state.y_previous[i])
    end

    # Update the function value and gradient
    state.f_x_previous, state.f_x = state.f_x, d.fg!(state.x, state.g)
    state.f_calls, state.g_calls = state.f_calls + 1, state.g_calls + 1

    false # don't force break
end
