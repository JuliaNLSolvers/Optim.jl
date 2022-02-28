# Translation from our variables to Nocedal and Wright's
# JMW's dx <=> NW's s
# JMW's dg <=> NW' y

struct BFGS{IL, L, H, T, TM} <: FirstOrderOptimizer
    alphaguess!::IL
    linesearch!::L
    initial_invH::H
    initial_stepnorm::T
    manifold::TM
end

Base.summary(::BFGS) = "BFGS"

"""
# BFGS
## Constructor
```julia
BFGS(; alphaguess = LineSearches.InitialStatic(),
       linesearch = LineSearches.HagerZhang(),
       initial_invH = x -> Matrix{eltype(x)}(I, length(x), length(x)),
       manifold = Flat())
```

## Description
The `BFGS` method implements the Broyden-Fletcher-Goldfarb-Shanno algorithm as
described in Nocedal and Wright (sec. 8.1, 1999) and the four individual papers
Broyden (1970), Fletcher (1970), Goldfarb (1970), and Shanno (1970). It is a
quasi-Newton method that updates an approximation to the Hessian using past
approximations as well as the gradient. See also the limited memory variant
`LBFGS` for an algorithm that is more suitable for high dimensional problems.

## References
 - Wright, S. J. and J. Nocedal (1999), Numerical optimization. Springer Science 35.67-68: 7.
 - Broyden, C. G. (1970), The convergence of a class of double-rank minimization algorithms, Journal of the Institute of Mathematics and Its Applications, 6: 76–90.
 - Fletcher, R. (1970), A New Approach to Variable Metric Algorithms, Computer Journal, 13 (3): 317–322,
 - Goldfarb, D. (1970), A Family of Variable Metric Updates Derived by Variational Means, Mathematics of Computation, 24 (109): 23–26,
 - Shanno, D. F. (1970), Conditioning of quasi-Newton methods for function minimization, Mathematics of Computation, 24 (111): 647–656.
"""
function BFGS(; alphaguess = LineSearches.InitialStatic(), # TODO: benchmark defaults
                linesearch = LineSearches.HagerZhang(),  # TODO: benchmark defaults
                initial_invH = nothing,
                initial_stepnorm = nothing,
                manifold::Manifold=Flat())
    BFGS(_alphaguess(alphaguess), linesearch, initial_invH, initial_stepnorm, manifold)
end

mutable struct BFGSState{Tx, Tm, T,G} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    g_previous::G
    f_x_previous::T
    dx::Tx
    dg::Tx
    u::Tx
    invH::Tm
    s::Tx
    @add_linesearch_fields()
end

function _init_identity_matrix(x::AbstractArray{T}, scale::T = T(1)) where {T}
    x_ = reshape(x, :)
    Id = x_ .* x_' .* false
    idxs = diagind(Id)
    @. @view(Id[idxs]) = scale * true
    return Id
end

function reset!(method, state::BFGSState, obj, x)
    n = length(x)
    T = eltype(x)
    retract!(method.manifold, x)
    value_gradient!(obj, x)
    project_tangent!(method.manifold, gradient(obj), x)

    if method.initial_invH === nothing
        if method.initial_stepnorm === nothing
            # Identity matrix of size n x n
            state.invH = _init_identity_matrix(x)
        else
            initial_scale = T(method.initial_stepnorm) * inv(norm(gradient(obj), Inf))
            state.invH = _init_identity_matrix(x, initial_scale)
        end
    else
        state.invH .= method.initial_invH(x)
    end
end

function initial_state(method::BFGS, options, d, initial_x::AbstractArray{T}) where T
    n = length(initial_x)
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)

    value_gradient!!(d, initial_x)

    project_tangent!(method.manifold, gradient(d), initial_x)

    if method.initial_invH === nothing
        if method.initial_stepnorm === nothing
            # Identity matrix of size n x n
            invH0 = _init_identity_matrix(initial_x)
        else
            initial_scale = T(method.initial_stepnorm) * inv(norm(gradient(d), Inf))
            invH0 = _init_identity_matrix(initial_x, initial_scale)
        end
    else
        invH0 = method.initial_invH(initial_x)
    end
    # Maintain a cache for line search results
    # Trace the history of states visited
    BFGSState(initial_x, # Maintain current state in state.x
              copy(initial_x), # Maintain previous state in state.x_previous
              copy(gradient(d)), # Store previous gradient in state.g_previous
              real(T)(NaN), # Store previous f in state.f_x_previous
              similar(initial_x), # Store changes in position in state.dx
              similar(initial_x), # Store changes in gradient in state.dg
              similar(initial_x), # Buffer stored in state.u
              invH0, # Store current invH in state.invH
              similar(initial_x), # Store current search direction in state.s
              @initial_linesearch()...)
end


function update_state!(d, state::BFGSState, method::BFGS)
    n = length(state.x)
    T = eltype(state.s)
    # Set the search direction
    # Search direction is the negative gradient divided by the approximate Hessian
    mul!(vec(state.s), state.invH, vec(gradient(d)))
    rmul!(state.s, T(-1))
    project_tangent!(method.manifold, state.s, state.x)

    # Maintain a record of the previous gradient
    copyto!(state.g_previous, gradient(d))

    # Determine the distance of movement along the search line
    # This call resets invH to initial_invH is the former in not positive
    # semi-definite
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Update current position
    state.dx .= state.alpha.*state.s
    state.x .= state.x .+ state.dx
    retract!(method.manifold, state.x)

    lssuccess == false # break on linesearch error
end

function update_h!(d, state, method::BFGS)
    n = length(state.x)
    # Measure the change in the gradient
    state.dg .= gradient(d) .- state.g_previous

    # Update the inverse Hessian approximation using Sherman-Morrison
    dx_dg = real(dot(state.dx, state.dg))
    if dx_dg > 0
        mul!(vec(state.u), state.invH, vec(state.dg))

        c1 = (dx_dg + real(dot(state.dg, state.u))) / (dx_dg' * dx_dg)
        c2 = 1 / dx_dg

        # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
        if(state.invH isa Array) # i.e. not a CuArray
            invH = state.invH; dx = state.dx; u = state.u;
            @inbounds for j in 1:n
                c1dxj = c1 * dx[j]'
                c2dxj = c2 * dx[j]'
                c2uj  = c2 *  u[j]'
                for i in 1:n
                    invH[i, j] = muladd(dx[i], c1dxj, muladd(-u[i], c2dxj, muladd(c2uj, -dx[i], invH[i, j])))
                end
            end
        else
            mul!(state.invH,vec(state.dx),vec(state.dx)', c1,1)
            mul!(state.invH,vec(state.u ),vec(state.dx)',-c2,1)
            mul!(state.invH,vec(state.dx),vec(state.u )',-c2,1)
        end
    end
end

function trace!(tr, d, state, iteration, method::BFGS, options, curr_time=time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["~inv(H)"] = copy(state.invH)
        dt["Current step size"] = state.alpha
    end
    g_norm = norm(gradient(d), Inf)
    update!(tr,
    iteration,
    value(d),
    g_norm,
    dt,
    options.store_trace,
    options.show_trace,
    options.show_every,
    options.callback)
end
