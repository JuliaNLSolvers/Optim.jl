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
function BFGS(; linesearch = LineSearches.hagerzhang!,
                initial_invH = x -> eye(eltype(x), length(x)),
                resetalpha = true)
    BFGS(linesearch, initial_invH, resetalpha)
end

type BFGSState{T,N,G}
    @add_generic_fields()
    x_previous::Array{T,N}
    g_previous::G
    f_x_previous::T
    dx::Array{T,N}
    dg::Array{T,N}
    u::Array{T,N}
    invH::Matrix{T}
    s::Array{T,N}
    @add_linesearch_fields()
end

function initial_state{T}(method::BFGS, options, d, initial_x::Array{T})
    n = length(initial_x)
    value_grad!(d, initial_x)
    # Maintain a cache for line search results
    # Trace the history of states visited
    BFGSState("BFGS",
              n,
              copy(initial_x), # Maintain current state in state.x
              similar(initial_x), # Maintain previous state in state.x_previous
              copy(gradient(d)), # Store previous gradient in state.g_previous
              T(NaN), # Store previous f in state.f_x_previous
              similar(initial_x), # Store changes in position in state.dx
              similar(initial_x), # Store changes in gradient in state.dg
              similar(initial_x), # Buffer stored in state.u
              method.initial_invH(initial_x), # Store current invH in state.invH
              similar(initial_x), # Store current search direction in state.s
              @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end


function update_state!{T}(d, state::BFGSState{T}, method::BFGS)
    # Set the search direction
    # Search direction is the negative gradient divided by the approximate Hessian
    A_mul_B!(state.s, state.invH, gradient(d))
    scale!(state.s, -1)

    # Maintain a record of the previous gradient
    copy!(state.g_previous, gradient(d))

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

    lssuccess == false # break on linesearch error
end

function update_h!(d, state, method::BFGS)
    # Measure the change in the gradient
    @simd for i in 1:state.n
        @inbounds state.dg[i] = gradient(d, i) - state.g_previous[i]
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
