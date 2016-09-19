immutable GradientDescent{T} <: Optimizer
    linesearch!::Function
    P::T
    precondprep!::Function
end

GradientDescent(; linesearch!::Function = hz_linesearch!,
                P = nothing, precondprep! = (P, x) -> nothing) =
                    GradientDescent(linesearch!, P, precondprep!)

type GradientDescentState{T}
    @add_generic_fields()
    x_previous::Array{T}
    g::Array{T}
    f_x_previous::T
    s::Array{T}
    @add_linesearch_fields()
end

function initial_state{T}(method::GradientDescent, options, d, initial_x::Array{T})
    g = similar(initial_x)
    f_x = d.fg!(initial_x, g)

    GradientDescentState("Gradient Descent",
                         length(initial_x),
                         copy(initial_x), # Maintain current state in state.x
                         f_x, # Store current f in state.f_x
                         1, # Track f calls in state.f_calls
                         1, # Track g calls in state.g_calls
                         0, # Track h calls in state.h_calls
                         0., # Track elapsed time
                         copy(initial_x), # Maintain current state in state.x_previous
                         g, # Store current gradient in state.g
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::GradientDescentState{T}, method::GradientDescent)
    # Search direction is always the negative preconditioned gradient
    method.precondprep!(method.P, state.x)
    A_ldiv_B!(state.s, method.P, state.g)
    @simd for i in 1:state.n
        @inbounds state.s[i] = -state.s[i]
    end
    # Refresh the line search cache
    dphi0 = vecdot(state.g, state.s)
    clear!(state.lsr)
    push!(state.lsr, zero(T), state.f_x, dphi0)

    # Determine the distance of movement along the search line
    state.alpha, f_update, g_update =
    method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls,
                       state.lsr, state.alpha, state.mayterminate)
    state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    LinAlg.axpy!(state.alpha, state.s, state.x)
    false # don't force break
end
