# Translation from our variables to Nocedal and Wright's
# JMW's dx <=> NW's s
# JMW's dg <=> NW' y
immutable BFGS{L<:Function, H<:Function} <: Optimizer
    linesearch!::L
    initial_invH::H
    resetalpha::Bool
end
#= uncomment for v0.8.0
BFGS(; linesearch = LineSearches.hagerzhang!, initial_invH = x -> eye(eltype(x), length(x))) =
  BFGS(linesearch, initial_invH)
=#
function BFGS(; linesearch! = nothing,
                linesearch = LineSearches.hagerzhang!,
              initial_invH = x -> eye(eltype(x), length(x)),
              resetalpha = true)
    linesearch = get_linesearch(linesearch!, linesearch)
    BFGS(linesearch, initial_invH, resetalpha)
end

type BFGSState{T}
    @add_generic_fields()
    x_previous::Array{T}
    g::Array{T}
    g_previous::Array{T}
    f_x_previous::T
    dx::Array{T}
    dg::Array{T}
    u::Array{T}
    invH::Array{T}
    s::Array{T}
    @add_linesearch_fields()
end

function initial_state{T}(method::BFGS, options, d, initial_x::Array{T})
    n = length(initial_x)
    g = similar(initial_x)
    f_x = d.fg!(initial_x, g)
    # Maintain a cache for line search results
    # Trace the history of states visited
    BFGSState("BFGS",
              n,
              copy(initial_x), # Maintain current state in state.x
              f_x, # Store current f in state.f_x
              1, # Track f calls in state.f_calls
              1, # Track g calls in state.g_calls
              0, # Track h calls in state.h_calls
              copy(initial_x), # Maintain current state in state.x_previous
              g, # Store current gradient in state.g
              copy(g), # Store previous gradient in state.g_previous
              T(NaN), # Store previous f in state.f_x_previous
              Array{T}(n), # Store changes in position in state.dx
              Array{T}(n), # Store changes in gradient in state.dg
              Array{T}(n), # Buffer stored in state.u
              method.initial_invH(initial_x), # Store current invH in state.invH
              Array{T}(n), # Store current search direction in state.s
              @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end


function update_state!{T}(d, state::BFGSState{T}, method::BFGS)
    # Set the search direction
    # Search direction is the negative gradient divided by the approximate Hessian
    A_mul_B!(state.s, state.invH, state.g)
    scale!(state.s, -1)

    # Determine the distance of movement along the search line
    # This call resets invH to initial_invH is the former in not positive
    # semi-definite
    lssuccess = perform_linesearch!(state, method, d)

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position
    @simd for i in 1:state.n
        @inbounds state.dx[i] = state.alpha * state.s[i]
        @inbounds state.x[i] = state.x[i] + state.dx[i]
    end

    # Maintain a record of the previous gradient
    copy!(state.g_previous, state.g)
    (lssuccess == false) # break on linesearch error
end

function update_h!(d, state, mehtod::BFGS)
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
end
