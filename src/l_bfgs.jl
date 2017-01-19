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

immutable LBFGS{T, L<:Function, Tprep<:Union{Function, Void}} <: Optimizer
    m::Int
    linesearch!::L
    P::T
    precondprep!::Tprep
    extrapolate::Bool
    snap2one::Tuple
end
#= uncomment for v0.8.0
LBFGS(; m::Integer = 10, linesearch = LineSearches.hagerzhang!,
                        P=nothing, precondprep = (P, x) -> nothing,
                        extrapolate::Bool=false, snap2one = (0.75, Inf)) =
      LBFGS(Int(m), linesearch, P, precondprep, extrapolate, snap2one)
=#

function LBFGS(; linesearch! = nothing,
                 m::Integer = 10,
                 linesearch = LineSearches.hagerzhang!,
                 P=nothing,
                 precondprep! = nothing,
                 precondprep = (P, x) -> nothing,
                 extrapolate::Bool=false,
                 snap2one = (0.75, Inf))

    linesearch = get_linesearch(linesearch!, linesearch)
    precondprep = get_precondprep(precondprep!, precondprep)
    LBFGS(Int(m), linesearch, P, precondprep, extrapolate, snap2one)
end

type LBFGSState{T}
    @add_generic_fields()
    x_previous::Array{T}
    g::Array{T}
    g_previous::Array{T}
    rho::Array{T}
    dx_history::Array{T}
    dg_history::Array{T}
    dx::Array{T}
    dg::Array{T}
    u::Array{T}
    f_x_previous::T
    twoloop_q
    twoloop_alpha
    pseudo_iteration::Int
    s::Array{T}
    @add_linesearch_fields()
end

function initial_state{T}(method::LBFGS, options, d, initial_x::Array{T})
    n = length(initial_x)
    g = similar(initial_x)
    f_x = d.fg!(initial_x, g)
    # Maintain a cache for line search results
    # Trace the history of states visited
    LBFGSState("L-BFGS",
              n,
              copy(initial_x), # Maintain current state in state.x
              f_x, # Store current f in state.f_x
              1, # Track f calls in state.f_calls
              1, # Track g calls in state.g_calls
              0, # Track h calls in state.h_calls
              copy(initial_x), # Maintain current state in state.x_previous
              g, # Store current gradient in state.g
              copy(g), # Store previous gradient in state.g_previous
              Array{T}(method.m), # state.rho
              Array{T}(n, method.m), # Store changes in position in state.dx_history
              Array{T}(n, method.m), # Store changes in gradient in state.dg_history
              Array{T}(n), # Buffer for new entry in state.dx_history
              Array{T}(n), # Buffer for new entry in state.dg_history
              Array{T}(n), # Buffer stored in state.u
              T(NaN), # Store previous f in state.f_x_previous
              Array{T}(n), #Buffer for use by twoloop
              Array{T}(method.m), #Buffer for use by twoloop
              0,
              Array{T}(n), # Store current search direction in state.s
              @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::LBFGSState{T}, method::LBFGS)
    lssuccess = true
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

    LineSearches.clear!(state.lsr)
    push!(state.lsr, zero(T), state.f_x, dphi0)

    # compute an initial guess for the linesearch based on
    # Nocedal/Wright, 2nd ed, (3.60)
    # TODO: this is a temporary fix, but should eventually be split off into
    #       a separate type and possibly live in LineSearches; see #294
    if method.extrapolate && state.pseudo_iteration > 1
        alphaguess = 2.0 * (state.f_x - state.f_x_previous) / dphi0
        alphaguess = max(alphaguess, state.alpha/4.0)  # not too much reduction
        # if alphaguess ≈ 1, then make it 1 (Newton-type behaviour)
        if method.snap2one[1] < alphaguess < method.snap2one[2]
            alphaguess = 1.0
        end
    else
        # without extrapolation use previous alpha (old behaviour)
        alphaguess = state.alpha
    end

    # Determine the distance of movement along the search line
    try
        state.alpha, f_update, g_update =
        method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls, state.lsr,
                           alphaguess, state.mayterminate)
        state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update
    catch ex
        if isa(ex, LineSearches.LineSearchException)
            lssuccess = false
            state.f_calls, state.g_calls = state.f_calls + ex.f_update, state.g_calls + ex.g_update
            state.alpha = ex.alpha
            Base.warn("Linesearch failed, using alpha = $(state.alpha) and exiting optimization.")
        else
            rethrow(ex)
        end
    end

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position
    @simd for i in 1:n
        @inbounds state.dx[i] = state.alpha * state.s[i]
        @inbounds state.x[i] = state.x[i] + state.dx[i]
    end

    # Save old f and g values to prepare for update_g! call
    state.f_x_previous = state.f_x
    copy!(state.g_previous, state.g)
    (lssuccess == false) # break on linesearch error
end


function update_h!(d, state, method::LBFGS)
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

end
