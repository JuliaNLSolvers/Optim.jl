# See p. 280 of Murphy's Machine Learning
# x_k1 = x_k - alpha * gr + mu * (x - x_previous)

immutable MomentumGradientDescent{L<:Function} <: Optimizer
    mu::Float64
    linesearch!::L
end

#= uncomment for v0.8.0
MomentumGradientDescent(; mu::Real = 0.01, linesearch = LineSearches.hagerzhang!) =
  MomentumGradientDescent(Float64(mu), linesearch)
=#

function MomentumGradientDescent(; mu::Real = 0.01, linesearch! = nothing,
                                   linesearch = LineSearches.hagerzhang!)
    linesearch = get_linesearch(linesearch!, linesearch)
    MomentumGradientDescent(Float64(mu), linesearch)
end

type MomentumGradientDescentState{T,N,G}
    @add_generic_fields()
    x_previous::Array{T,N}
    g::G
    f_x_previous::T
    s::Array{T,N}
    @add_linesearch_fields()
end

function initial_state{T}(method::MomentumGradientDescent, options, d, initial_x::Array{T})
    g = similar(initial_x)
    f_x = d.fg!(initial_x, g)

    MomentumGradientDescentState("Momentum Gradient Descent",
                         length(initial_x),
                         copy(initial_x), # Maintain current state in state.x
                         f_x, # Store current f in state.f_x
                         1, # Track f calls in state.f_calls
                         1, # Track g calls in state.g_calls
                         0, # Track h calls in state.h_calls
                         copy(initial_x), # Maintain previous state in state.x_previous
                         g, # Store current gradient in state.g
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::MomentumGradientDescentState{T}, method::MomentumGradientDescent)
    # Search direction is always the negative gradient
    @simd for i in 1:state.n
        @inbounds state.s[i] = -state.g[i]
    end

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Update current position
    @simd for i in 1:state.n
        # Need to move x into x_previous while using x_previous and creating "x_new"
        @inbounds tmp = state.x_previous[i]
        @inbounds state.x_previous[i] = state.x[i]
        @inbounds state.x[i] = state.x[i] + state.alpha * state.s[i] + method.mu * (state.x[i] - tmp)
    end
    lssuccess == false # break on linesearch error
end
