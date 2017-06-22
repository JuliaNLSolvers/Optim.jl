# L should be function or any other callable
struct GradientDescent{L, T, Tprep<:Union{Function, Void}} <: Optimizer
    linesearch!::L
    P::T
    precondprep!::Tprep
end

#= uncomment for v0.8.0
GradientDescent(; linesearch = LineSearches.HagerZhang(),
                P = nothing, precondprep = (P, x) -> nothing) =
                    GradientDescent(linesearch, P, precondprep)
=#

Base.summary(::GradientDescent) = "Gradient Descent"

function GradientDescent(; linesearch = LineSearches.HagerZhang(),
                           P = nothing,
                           precondprep = (P, x) -> nothing)
    GradientDescent(linesearch, P, precondprep)
end

mutable struct GradientDescentState{T,N}
    x::Array{T,N}
    x_previous::Array{T,N}
    f_x_previous::T
    s::Array{T,N}
    @add_linesearch_fields()
end

function initial_state{T}(method::GradientDescent, options, d, initial_x::Array{T})
    value_gradient!(d, initial_x)

    GradientDescentState(copy(initial_x), # Maintain current state in state.x
                         similar(initial_x), # Maintain previous state in state.x_previous
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::GradientDescentState{T}, method::GradientDescent)
    n = length(state.x)
    # Search direction is always the negative preconditioned gradient
    method.precondprep!(method.P, state.x)
    A_ldiv_B!(state.s, method.P, gradient(d))
    @simd for i in 1:n
        @inbounds state.s[i] = -state.s[i]
    end

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    LinAlg.axpy!(state.alpha, state.s, state.x)
    lssuccess == false # break on linesearch error
end

function assess_convergence(state::GradientDescentState, d, options)
  default_convergence_assessment(state, d, options)
end

function trace!(tr, d, state, iteration, method::GradientDescent, options)
  common_1order_trace!(tr, d, state, iteration, method, options)
end
