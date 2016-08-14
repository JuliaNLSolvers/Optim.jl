# Translation from our variables to Nocedal and Wright's
# JMW's dx <=> NW's s
# JMW's dg <=> NW' y
immutable BFGS <: Optimizer
    linesearch!::Function
    initial_invH::Function
end

method_string(method::BFGS) = "BFGS"

BFGS(; linesearch!::Function = hz_linesearch!, initial_invH = x -> eye(eltype(x), length(x))) =
  BFGS(linesearch!, initial_invH)

function trace!(tr, state, iteration, method::BFGS, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g)
        dt["~inv(H)"] = copy(state.invH)
        dt["Current step size"] = state.alpha
    end
    g_norm = vecnorm(state.g, Inf)
    update!(tr,
            iteration,
            state.f_x,
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end

type BFGSState{T}
    n::Int64
    x::Array{T}
    x_previous::Array{T}
    g::Array{T}
    g_previous::Array{T}
    dx::Array{T}
    dg::Array{T}
    s::Array{T}
    u::Array{T}
    I::Array{T}
    invH::Array{T}
    f_x::T
    f_x_previous::T
    x_ls::Array{T}
    g_ls::Array{T}
    alpha::T
    mayterminate::Bool
    f_calls::Int64
    g_calls::Int64
    lsr
end

function initialize_state{T}(method::BFGS, options, d, initial_x::Array{T})
    n = length(initial_x)
    g = Array(T, n)
    f_x = d.fg!(initial_x, g)
    invH = method.initial_invH(initial_x)
    # Maintain a cache for line search results
    # Trace the history of states visited
    BFGSState(n,
              copy(initial_x), # Maintain current state in state.x
              copy(initial_x), # Maintain current state in state.x_previous
              g, # Store current gradient in state.g
              copy(g), # Store previous gradient in state.g_previous
              Array{T}(n), # Store changes in position in state.dx
              Array{T}(n), # Store changes in gradient in state.dg
              Array{T}(n), # Store current search direction in state.s
              Array{T}(n), # Buffer stored in state.u
              eye(T, size(invH)...),
              invH, # Store current invH in state.invH
              f_x, # Store current f in state.f_x
              T(NaN), # Store previous f in state.f_x_previous
              similar(initial_x), # Buffer of x for line search in state.x_ls
              similar(initial_x), # Buffer of g for line search in state.g_ls
              alphainit(one(T), initial_x, g, f_x), # Keep track of step size in state.alpha
              false, # state.mayterminate
              1, # Track f calls in state.f_calls
              1, # Track g calls in state.g_calls
              LineSearchResults(T)) # Maintain a cache for line search results in state.lsr
end


function update!{T}(d, state::BFGSState{T}, method::BFGS)

    # Set the search direction
    # Search direction is the negative gradient divided by the approximate Hessian
    A_mul_B!(state.s, state.invH, state.g)
    scale!(state.s, -1)

    # Refresh the line search cache
    dphi0 = vecdot(state.g, state.s)
    # If invH is not positive definite, reset it to I
    if dphi0 > 0.0
        copy!(state.invH, state.I)
        @simd for i in 1:state.n
            @inbounds state.s[i] = -state.g[i]
        end
        dphi0 = vecdot(state.g, state.s)
    end
    clear!(state.lsr)
    push!(state.lsr, zero(T), state.f_x, dphi0)

    # Determine the distance of movement along the search line
    state.alpha, f_update, g_update =
      method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls, state.lsr, state.alpha, state.mayterminate)
    state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position
    @simd for i in 1:state.n
        @inbounds state.dx[i] = state.alpha * state.s[i]
        @inbounds state.x[i] = state.x[i] + state.dx[i]
    end

    # Maintain a record of the previous gradient
    copy!(state.g_previous, state.g)

    # Update the function value and gradient
    state.f_x_previous, state.f_x = state.f_x, d.fg!(state.x, state.g)
    state.f_calls, state.g_calls = state.f_calls + 1, state.g_calls + 1

    # Measure the change in the gradient
    @simd for i in 1:state.n
        @inbounds state.dg[i] = state.g[i] - state.g_previous[i]
    end

    # Update the inverse Hessian approximation using Sherman-Morrison
    dx_dg = vecdot(state.dx, state.dg)
    if dx_dg == 0.0
        return true # force stop
    end
    A_mul_B!(state.u, state.invH, state.dg)

    c1 = (dx_dg + vecdot(state.dg, state.u)) / (dx_dg * dx_dg)
    c2 = 1 / dx_dg

    # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
    for i in 1:state.n
        @simd for j in 1:state.n
            @inbounds state.invH[i, j] += c1 * state.dx[i] * state.dx[j] - c2 * (state.u[i] * state.dx[j] + state.u[j] * state.dx[i])
        end
    end
    false # don't force stop
end
