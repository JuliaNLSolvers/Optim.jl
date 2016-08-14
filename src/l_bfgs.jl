# Notational note
# JMW's dx_history <=> NW's S
# JMW's dg_history <=> NW's Y

# Here alpha is a cache that parallels betas
# It is not the step-size
# q is also a cache
function twoloop!(s::Vector,
                  gr::Vector,
                  rho::Vector,
                  dx_history::Matrix,
                  dg_history::Matrix,
                  m::Integer,
                  pseudo_iteration::Integer,
                  alpha::Vector,
                  q::Vector,
                  precon)
    # Count number of parameters
    n = length(s)

    # Determine lower and upper bounds for loops
    lower = pseudo_iteration - m
    upper = pseudo_iteration - 1

    # Copy gr into q for backward pass
    copy!(q, gr)

    # Backward pass
    for index in upper:-1:lower
        if index < 1
            continue
        end
        i = mod1(index, m)
        @inbounds alpha[i] = rho[i] * vecdot(view(dx_history, :, i), q)
        @simd for j in 1:n
            @inbounds q[j] -= alpha[i] * dg_history[j, i]
        end
    end

    # Copy q into s for forward pass
    # apply preconditioner if precon != nothing
    # (Note: preconditioner update was done outside of this function)
    A_ldiv_B!(s, precon, q)

    # Forward pass
    for index in lower:1:upper
        if index < 1
            continue
        end
        i = mod1(index, m)
        @inbounds beta = rho[i] * vecdot(view(dg_history, :, i), s)
        @simd for j in 1:n
            @inbounds s[j] += dx_history[j, i] * (alpha[i] - beta)
        end
    end

    # Negate search direction
    scale!(s, -1)

    return
end

immutable LBFGS{T} <: Optimizer
    m::Int
    linesearch!::Function
    P::T
    precondprep!::Function
end

LBFGS(; m::Integer = 10, linesearch!::Function = hz_linesearch!,
      P=nothing, precondprep! = (P, x) -> nothing) =
    LBFGS(Int(m), linesearch!, P, precondprep!)

method_string(method::LBFGS) = "L-BFGS"

function trace!(tr, state, iteration, method::LBFGS, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(x)
        dt["g(x)"] = copy(gr)
        dt["Current step size"] = alpha
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

type LBFGSState{T}
    n::Int64
    x::Array{T}
    x_previous::Array{T}
    g::Array{T}
    g_previous::Array{T}
    rho::Array{T}
    dx_history::Array{T}
    dg_history::Array{T}
    dx::Array{T}
    dg::Array{T}
    s::Array{T}
    u::Array{T}
    f_x::T
    f_x_previous::T
    x_ls::Array{T}
    g_ls::Array{T}
    alpha::T
    twoloop_q
    twoloop_alpha
    mayterminate::Bool
    pseudo_iteration::Int64
    f_calls::Int64
    g_calls::Int64
    lsr
end

function initialize_state{T}(method::LBFGS, options, d, initial_x::Array{T})
    n = length(initial_x)
    g = Array(T, n)
    f_x = d.fg!(initial_x, g)
    # Maintain a cache for line search results
    # Trace the history of states visited
    LBFGSState(n,
              copy(initial_x), # Maintain current state in state.x
              copy(initial_x), # Maintain current state in state.x_previous
              g, # Store current gradient in state.g
              copy(g), # Store previous gradient in state.g_previous
              Array{T}(method.m), # state.rho
              Array{T}(n, method.m), # Store changes in position in state.dx_history
              Array{T}(n, method.m), # Store changes in gradient in state.dg_history
              Array{T}(n), # Buffer for new entry in state.dx_history
              Array{T}(n), # Buffer for new entry in state.dg_history
              Array{T}(n), # Store current search direction in state.s
              Array{T}(n), # Buffer stored in state.u
              f_x, # Store current f in state.f_x
              T(NaN), # Store previous f in state.f_x_previous
              similar(initial_x), # Buffer of x for line search in state.x_ls
              similar(initial_x), # Buffer of g for line search in state.g_ls
              alphainit(one(T), initial_x, g, f_x), # Keep track of step size in state.alpha
              Array{T}(n), #Buffer for use by twoloop
              Array{T}(method.m), #Buffer for use by twoloop
              false, # state.mayterminate
              0,
              1, # Track f calls in state.f_calls
              1, # Track g calls in state.g_calls
              LineSearchResults(T)) # Maintain a cache for line search results in state.lsr
end

function update!{T}(d, state::LBFGSState{T}, method::LBFGS)
    n = state.n
    # Increment the number of steps we've had to perform
    state.pseudo_iteration += 1

    # update the preconditioner
    method.precondprep!(method.P, state.x)

    # Determine the L-BFGS search direction # FIXME just pass state and method?
    twoloop!(state.s, state.g, state.rho, state.dx_history, state.dg_history,
             method.m, state.pseudo_iteration,
             state.twoloop_alpha, state.twoloop_q, method.P)

    # Refresh the line search cache
    dphi0 = vecdot(state.g, state.s)
    if dphi0 > 0.0
        state.pseudo_iteration = 1
        @simd for i in 1:n
            @inbounds state.s[i] = -state.g[i]
        end
        dphi0 = vecdot(state.g, state.s)
    end

    clear!(state.lsr)
    push!(state.lsr, zero(T), state.f_x, dphi0)

    # Determine the distance of movement along the search line
    state.alpha, f_update, g_update =
      method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls, state.lsr,
                     state.alpha, state.mayterminate)
    state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position
    @simd for i in 1:n
        @inbounds state.dx[i] = state.alpha * state.s[i]
        @inbounds state.x[i] = state.x[i] + state.dx[i]
    end

    # Maintain a record of the previous gradient
    copy!(state.g_previous, state.g)

    # Update the function value and gradient
    state.f_x_previous = state.f_x
    state.f_x = d.fg!(state.x, state.g)
    state.f_calls, state.g_calls = state.f_calls + 1, state.g_calls + 1

    # Measure the change in the gradient
    @simd for i in 1:state.n
        @inbounds state.dg[i] = state.g[i] - state.g_previous[i]
    end

    # Update the L-BFGS history of positions and gradients
    rho_iteration = 1 / vecdot(state.dx, state.dg)
    if isinf(rho_iteration)
        # TODO: Introduce a formal error? There was a warning here previously
        return true
    end
    state.dx_history[:, mod1(state.pseudo_iteration, method.m)] = state.dx
    state.dg_history[:, mod1(state.pseudo_iteration, method.m)] = state.dg
    state.rho[mod1(state.pseudo_iteration, method.m)] = rho_iteration

    false
end
