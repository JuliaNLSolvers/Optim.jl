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

type LBFGSState{T,N,M}
    @add_generic_fields()
    x_previous::Array{T,N}
    g::Array{T,N}
    g_previous::Array{T,N}
    rho::Array{T,N}
    dx_history::Array{T,M}
    dg_history::Array{T,M}
    dx::Array{T,N}
    dg::Array{T,N}
    u::Array{T,N}
    f_x_previous::T
    twoloop_q
    twoloop_alpha
    pseudo_iteration::Int
    s::Array{T,N}
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
              similar(initial_x), # Maintain previous state in state.x_previous
              g, # Store current gradient in state.g
              similar(g), # Store previous gradient in state.g_previous
              Array{T}(method.m), # state.rho
              Array{T}(n, method.m), # Store changes in position in state.dx_history
              Array{T}(n, method.m), # Store changes in gradient in state.dg_history
              similar(initial_x), # Buffer for new entry in state.dx_history
              similar(initial_x), # Buffer for new entry in state.dg_history
              similar(initial_x), # Buffer stored in state.u
              T(NaN), # Store previous f in state.f_x_previous
              similar(initial_x), #Buffer for use by twoloop
              Array{T}(method.m), #Buffer for use by twoloop
              0,
              similar(initial_x), # Store current search direction in state.s
              @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::LBFGSState{T}, method::LBFGS)
    n = state.n
    # Increment the number of steps we've had to perform
    state.pseudo_iteration += 1

    # update the preconditioner
    method.precondprep!(method.P, state.x)

    # Determine the L-BFGS search direction # FIXME just pass state and method?
    twoloop!(state.s, state.g, state.rho, state.dx_history, state.dg_history,
             method.m, state.pseudo_iteration,
             state.twoloop_alpha, state.twoloop_q, method.P)


    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

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
    lssuccess == false # break on linesearch error
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
