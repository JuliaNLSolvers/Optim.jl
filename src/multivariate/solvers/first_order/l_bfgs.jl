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

# L should be function or any other callable
immutable LBFGS{T, L, Tprep<:Union{Function, Void}} <: Optimizer
    m::Int
    linesearch!::L
    P::T
    precondprep!::Tprep
    extrapolate::Bool
    snap2one::Tuple
end
#= uncomment for v0.8.0
LBFGS(; m::Integer = 10, linesearch = LineSearches.HagerZhang(),
                        P=nothing, precondprep = (P, x) -> nothing,
                        extrapolate::Bool=false, snap2one = (0.75, Inf)) =
      LBFGS(Int(m), linesearch, P, precondprep, extrapolate, snap2one)
=#

function LBFGS(; m::Integer = 10,
                 linesearch = LineSearches.HagerZhang(),
                 P=nothing,
                 precondprep = (P, x) -> nothing,
                 extrapolate::Bool=false,
                 snap2one = (0.75, Inf))

    LBFGS(Int(m), linesearch, P, precondprep, extrapolate, snap2one)
end

Base.summary(::LBFGS) = "L-BFGS"

type LBFGSState{T,N,M,G}
    x::Array{T,N}
    x_previous::Array{T,N}
    g_previous::G
    rho::Vector{T}
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
    value_gradient!(d, initial_x)
    LBFGSState(copy(initial_x), # Maintain current state in state.x
              similar(initial_x), # Maintain previous state in state.x_previous
              similar(gradient(d)), # Store previous gradient in state.g_previous
              Vector{T}(method.m), # state.rho
              Matrix{T}(n, method.m), # Store changes in position in state.dx_history
              Matrix{T}(n, method.m), # Store changes in gradient in state.dg_history
              similar(initial_x), # Buffer for new entry in state.dx_history
              similar(initial_x), # Buffer for new entry in state.dg_history
              similar(initial_x), # Buffer stored in state.u
              T(NaN), # Store previous f in state.f_x_previous
              similar(initial_x), #Buffer for use by twoloop
              Vector{T}(method.m), #Buffer for use by twoloop
              0,
              similar(initial_x), # Store current search direction in state.s
              @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::LBFGSState{T}, method::LBFGS)
    n = length(state.x)
    # Increment the number of steps we've had to perform
    state.pseudo_iteration += 1

    # update the preconditioner
    method.precondprep!(method.P, state.x)

    # Determine the L-BFGS search direction # FIXME just pass state and method?
    twoloop!(vec(state.s), vec(gradient(d)), vec(state.rho), state.dx_history, state.dg_history,
             method.m, state.pseudo_iteration,
             state.twoloop_alpha, vec(state.twoloop_q), method.P)

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)
    # Save old f and g values to prepare for update_g! call
    f_x_prev = value(d)
    copy!(state.g_previous, gradient(d))

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # This has to come after perform_linesearch since alphaguess! uses state.f_x_previous from the prior iteration
    state.f_x_previous = f_x_prev

    # Update current position
    state.dx .= state.alpha .* state.s
    state.x .= state.x .+ state.dx

    lssuccess == false # break on linesearch error
end


function update_h!(d, state, method::LBFGS)
    n = length(state.x)
    # Measure the change in the gradient
    @simd for i in 1:n
        @inbounds state.dg[i] = gradient(d, i) - state.g_previous[i]
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

function assess_convergence(state::LBFGSState, d, options)
  default_convergence_assessment(state, d, options)
end


function trace!(tr, d, state, iteration, method::LBFGS, options)
  common_1order_trace!(tr, d, state, iteration, method, options)
end
