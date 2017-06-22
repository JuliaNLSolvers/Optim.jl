# See p. 280 of Murphy's Machine Learning
# x_k1 = x_k - alpha * gr + mu * (x - x_previous)

# L should be function or any other callable
immutable MomentumGradientDescent{L} <: Optimizer
    mu::Float64
    linesearch!::L
end

#= uncomment for v0.8.0
MomentumGradientDescent(; mu::Real = 0.01, linesearch = LineSearches.HagerZhang()) =
  MomentumGradientDescent(Float64(mu), linesearch)
=#

Base.summary(::MomentumGradientDescent) = "Momentum Gradient Descent"

function MomentumGradientDescent(; mu::Real = 0.01, linesearch = LineSearches.HagerZhang())
    MomentumGradientDescent(Float64(mu), linesearch)
end

type MomentumGradientDescentState{T,N}
    x::Array{T,N}
    x_previous::Array{T,N}
    f_x_previous::T
    s::Array{T,N}
    @add_linesearch_fields()
end

function initial_state{T}(method::MomentumGradientDescent, options, d, initial_x::Array{T})
    value_gradient!(d, initial_x)

    MomentumGradientDescentState(copy(initial_x), # Maintain current state in state.x
                         copy(initial_x), # Maintain previous state in state.x_previous
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::MomentumGradientDescentState{T}, method::MomentumGradientDescent)
    n = length(state.x)
    # Search direction is always the negative gradient
    @simd for i in 1:n
        @inbounds state.s[i] = -gradient(d, i)
    end

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Update current position
    @simd for i in 1:n
        # Need to move x into x_previous while using x_previous and creating "x_new"
        @inbounds tmp = state.x_previous[i]
        @inbounds state.x_previous[i] = state.x[i]
        @inbounds state.x[i] = state.x[i] + state.alpha * state.s[i] + method.mu * (state.x[i] - tmp)
    end
    lssuccess == false # break on linesearch error
end

function assess_convergence(state::MomentumGradientDescentState, d, options)
  default_convergence_assessment(state, d, options)
end

function trace!(tr, d, state, iteration, method::MomentumGradientDescent, options)
  common_1order_trace!(tr, d, state, iteration, method, options)
end
