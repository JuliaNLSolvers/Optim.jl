# Translation from our variables to Nocedal and Wright's
# JMW's dx <=> NW's s
# JMW's dg <=> NW' y

struct BFGS{IL,L,H,T,TM,IS,SS} <: FirstOrderOptimizer
    alphaguess!::IL
    linesearch!::L
    initial_invH::H
    initial_stepnorm::T
    manifold::TM
    initial_scaling::IS
    self_scaling::SS
end

abstract type InitialScaling end
abstract type SelfScaling end

"""
    ShannoPhua()

`initial_scaling` rule for `BFGS`: just before the first update, replace the initial
inverse Hessian by `(s'y / y'y) I`, with `s` the step and `y` the change in gradient
(Shanno and Phua, 1978).
"""
struct ShannoPhua <: InitialScaling end

"""
    OrenLuenberger()

`self_scaling` rule for `BFGS`: before an update, scale the Hessian approximation
`B` by `τ = ρ = y's / s'Bs` (Oren and Luenberger, 1974).
"""
struct OrenLuenberger <: SelfScaling end

"""
    AlBaali(; cutoff = 0.5)

`self_scaling` rule for `BFGS`: before an update, scale the Hessian approximation
`B` by `τ = min(1, ρ)`, `ρ = y's / s'Bs`, and leave it unchanged when `ρ < cutoff`. This
is the restricted self-scaling of Al-Baali (1998) with the cutoff of Al-Baali and Khalfan
(2009). It only ever enlarges the inverse Hessian.
"""
struct AlBaali{T<:Real} <: SelfScaling
    cutoff::T
end
AlBaali(; cutoff::Real = 0.5) = AlBaali(cutoff)

_initial_scale(::ShannoPhua, s, y, sy) = sy / real(dot(y, y))
_self_scale(::OrenLuenberger, ρ) = ρ
_self_scale(r::AlBaali, ρ) = r.cutoff <= ρ < 1 ? ρ : one(ρ)

Base.summary(io::IO, ::BFGS) = print(io, "BFGS")

"""
# BFGS
## Constructor
```julia
BFGS(; alphaguess = LineSearches.InitialStatic(),
       linesearch = LineSearches.HagerZhang(),
       initial_invH = nothing,
       initial_stepnorm = 1.0,
       manifold = Flat(),
       initial_scaling = Optim.ShannoPhua(),
       self_scaling = Optim.AlBaali())
```

## Description
The `BFGS` method implements the Broyden-Fletcher-Goldfarb-Shanno algorithm as
described in Nocedal and Wright (sec. 8.1, 1999) and the four individual papers
Broyden (1970), Fletcher (1970), Goldfarb (1970), and Shanno (1970). It is a
quasi-Newton method that updates an approximation to the Hessian using past
approximations as well as the gradient. See also the limited memory variant
`LBFGS` for an algorithm that is more suitable for high dimensional problems.

The inverse Hessian approximation starts from the identity matrix, or from
`initial_invH(x0)` if given. `initial_stepnorm = z` divides the identity by the sup-norm
of the initial gradient and multiplies it by `z`, so the first trial step has sup-norm `z`;
`initial_stepnorm = nothing` keeps the identity, and the first trial step is then `-g`.

The defaults scale the inverse Hessian: a first step of sup-norm 1, `ShannoPhua` before the
first update and `AlBaali` after it. `BFGS(initial_stepnorm = nothing, initial_scaling =
nothing, self_scaling = nothing)` is the unscaled method that was the default up to
Optim 2.

`initial_scaling = Optim.ShannoPhua()` replaces the starting matrix by `(s'y / y'y) I`
just before the first update, and again after a reset of the search direction (Shanno and
Phua, 1978). Here `s` is the step and `y` the change in gradient. The update never changes
`w'Hu` for `w`, `u` orthogonal to the step, so the scale of the starting matrix persists in
every direction not yet explored. It is ignored when `initial_invH` is given; pass
`initial_scaling = nothing` to keep the identity.

`self_scaling` scales the Hessian approximation `B` by a factor `τ` before every other
update, that is, the inverse Hessian by `1/τ`. `Optim.OrenLuenberger()` uses
`τ = y's / s'Bs` (Oren and Luenberger, 1974). `Optim.AlBaali()` uses `τ = min(1, y's / s'Bs)`
and no scaling when `y's / s'Bs < 1/2` (Al-Baali, 1998; Al-Baali and Khalfan, 2009), so it
only ever enlarges the inverse Hessian. That repairs the case where `ShannoPhua` alone sets
it too small: on ill-conditioned quadratics `ShannoPhua` alone can take several times more
iterations than no scaling, and together with `AlBaali` fewer. `self_scaling` also applies
when `initial_invH` is given; pass `self_scaling = nothing` to update a supplied matrix
unscaled.

## References
 - Wright, S. J. and J. Nocedal (1999), Numerical optimization. Springer Science 35.67-68: 7.
 - Broyden, C. G. (1970), The convergence of a class of double-rank minimization algorithms, Journal of the Institute of Mathematics and Its Applications, 6: 76–90.
 - Fletcher, R. (1970), A New Approach to Variable Metric Algorithms, Computer Journal, 13 (3): 317–322,
 - Goldfarb, D. (1970), A Family of Variable Metric Updates Derived by Variational Means, Mathematics of Computation, 24 (109): 23–26,
 - Shanno, D. F. (1970), Conditioning of quasi-Newton methods for function minimization, Mathematics of Computation, 24 (111): 647–656.
 - Shanno, D. F. and K. H. Phua (1978), Matrix conditioning and nonlinear optimization, Mathematical Programming, 14: 149–160.
 - Oren, S. S. and D. G. Luenberger (1974), Self-scaling variable metric (SSVM) algorithms, Part I, Management Science, 20 (5): 845–862.
 - Al-Baali, M. (1998), Numerical experience with a class of self-scaling quasi-Newton algorithms, Journal of Optimization Theory and Applications, 96 (3): 533–553.
 - Al-Baali, M. and H. Khalfan (2009), A combined class of self-scaling and modified quasi-Newton methods, Optimization Online preprint 2334.
"""
function BFGS(;
    alphaguess = LineSearches.InitialStatic(), # TODO: benchmark defaults
    linesearch = LineSearches.HagerZhang(),  # TODO: benchmark defaults
    initial_invH = nothing,
    initial_stepnorm = 1.0,
    manifold::Manifold = Flat(),
    initial_scaling::Union{Nothing,InitialScaling} = ShannoPhua(),
    self_scaling::Union{Nothing,SelfScaling} = AlBaali(),
)
    BFGS(_alphaguess(alphaguess), linesearch, initial_invH, initial_stepnorm, manifold,
         initial_scaling, self_scaling)
end

mutable struct BFGSState{Tx,Tm,T,G} <: AbstractOptimizerState
    x::Tx
    g_x::G
    f_x::T
    x_previous::Tx
    g_x_previous::G
    f_x_previous::T
    dx::Tx
    dg::Tx
    u::Tx
    invH::Tm
    s::Tx
    # Trial iterate produced by update_state! / update_fgh!. Committed to
    # state.x / state.g_x / state.f_x by accept_step! once validated.
    x_candidate::Tx
    g_candidate::G
    f_candidate::T
    # true until the first update after the initial matrix was set (initial_scaling)
    rescale_invH::Bool
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
    # Update function value and gradient
    copyto!(state.x, x)
    retract!(method.manifold, state.x)
    f_x, g_x = NLSolversBase.value_gradient!(obj, state.x)
    copyto!(state.g_x, g_x)
    project_tangent!(method.manifold, state.g_x, state.x)
    state.f_x = f_x

    # Delete history
    fill!(state.x_previous, NaN)
    fill!(state.g_x_previous, NaN)
    state.f_x_previous = oftype(state.f_x_previous, NaN)

    # Update approximation of inverse Hessian
    if method.initial_invH === nothing
        if method.initial_stepnorm === nothing
            # Identity matrix of size n x n
            state.invH = _init_identity_matrix(x)
        else
            T = eltype(state.invH)
            initial_scale = T(method.initial_stepnorm) * inv(norm(g_x, Inf))
            state.invH = _init_identity_matrix(x, initial_scale)
        end
    else
        state.invH .= method.initial_invH(x)
    end
    state.rescale_invH = true

    return nothing
end

function initial_state(method::BFGS, ::Options, d, x0::AbstractArray)
    # Compute function value and gradient
    x0 = copy(x0)
    retract!(method.manifold, x0)
    f_x, g_x = NLSolversBase.value_gradient!(d, x0)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, x0)

    # Initialize approximation of inverse Hessian
    if method.initial_invH === nothing
        if method.initial_stepnorm === nothing
            # Identity matrix of size n x n
            invH0 = _init_identity_matrix(x0)
        else
            T = eltype(g_x)
            initial_scale = T(method.initial_stepnorm) * inv(norm(g_x, Inf))
            invH0 = _init_identity_matrix(x0, initial_scale)
        end
    else
        invH0 = method.initial_invH(x0)
    end

    # Maintain a cache for line search results
    # Trace the history of states visited
    BFGSState(
        x0, # Maintain current state in state.x
        g_x, # Maintain current gradient in state.g_x
        f_x, # Maintain current f in state.f_x
        fill!(similar(x0), NaN), # Maintain previous state in state.x_previous
        fill!(similar(g_x), NaN), # Store previous gradient in state.g_x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        fill!(similar(x0), NaN), # Store changes in position in state.dx
        fill!(similar(x0), NaN), # Store changes in gradient in state.dg
        fill!(similar(x0), NaN), # Buffer stored in state.u
        invH0, # Store current invH in state.invH
        fill!(similar(x0), NaN), # Store current search direction in state.s
        fill!(similar(x0), NaN), # Trial iterate in state.x_candidate
        fill!(similar(g_x), NaN), # Trial gradient in state.g_candidate
        oftype(f_x, NaN), # Trial f value in state.f_candidate
        true, # rescale_invH
        @initial_linesearch()...,
    )
end


function update_state!(d, state::BFGSState, method::BFGS)
    # Set the search direction
    # Search direction is the negative gradient divided by the approximate Hessian
    mul!(vec(state.s), state.invH, vec(state.g_x))
    rmul!(state.s, eltype(state.s)(-1))
    project_tangent!(method.manifold, state.s, state.x)

    # Determine the distance of movement along the search line
    # This call resets invH to initial_invH if the former is not positive
    # semi-definite
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Propose trial iterate (do NOT mutate state.x; accept_step! commits)
    copyto!(state.x_candidate, state.x_ls)
    retract!(method.manifold, state.x_candidate)
    # The displacement of the proposed step, which a retraction can bend away from alpha*s
    state.dx .= state.x_candidate .- state.x

    return !lssuccess # break on linesearch error
end

function update_fgh!(d, state::BFGSState, method::BFGS)
    f_c, g_c = NLSolversBase.value_gradient!(d, state.x_candidate)
    copyto!(state.g_candidate, g_c)
    project_tangent!(method.manifold, state.g_candidate, state.x_candidate)
    state.f_candidate = f_c
    return nothing
end

function accept_step!(d, state::BFGSState, method::BFGS, options)
    if !isfinite(state.f_candidate) ||
       !all(isfinite, state.g_candidate) ||
       !all(isfinite, state.x_candidate)
        return false
    end

    # Commit candidates.
    copyto!(state.x, state.x_candidate)
    copyto!(state.g_x, state.g_candidate)
    state.f_x = state.f_candidate

    (; invH, dx, dg, u) = state

    # Measure the change in the gradient
    dg .= state.g_x .- state.g_x_previous

    # Update the inverse Hessian approximation using Sherman-Morrison
    dx_dg = real(dot(dx, dg))
    if dx_dg > 0
        if method.initial_scaling !== nothing && method.initial_invH === nothing && state.rescale_invH
            γ = _initial_scale(method.initial_scaling, dx, dg, dx_dg)
            invH .= _init_identity_matrix(state.x, convert(eltype(state.x), γ))
            state.rescale_invH = false
        elseif method.self_scaling !== nothing
            # B <- τ B, so invH <- invH / τ. The step is dx = α d with B d = -g, so s'Bs = -α dx'g.
            sBs = -state.alpha * real(dot(dx, state.g_x_previous))
            if sBs > 0
                τ = _self_scale(method.self_scaling, dx_dg / sBs)
                isone(τ) || rmul!(invH, inv(τ))
            end
        end
        mul!(vec(u), invH, vec(dg))

        c1 = (dx_dg + real(dot(dg, u))) / abs2(dx_dg)
        c2 = 1 / dx_dg

        # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
        if (invH isa Array) # i.e. not a CuArray
            n = length(dx)
            @inbounds for j = 1:n
                c1dxj = c1 * dx[j]'
                c2dxj = c2 * dx[j]'
                c2uj = c2 * u[j]'
                for i = 1:n
                    invH[i, j] = muladd(
                        dx[i],
                        c1dxj,
                        muladd(-u[i], c2dxj, muladd(c2uj, -dx[i], invH[i, j])),
                    )
                end
            end
        else
            mul!(invH, vec(dx), vec(dx)', c1, 1)
            mul!(invH, vec(u), vec(dx)', -c2, 1)
            mul!(invH, vec(dx), vec(u)', -c2, 1)
        end
    end
    return true
end

function trace!(tr, d, state::BFGSState, iteration::Integer, method::BFGS, options::Options, curr_time = time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g_x)
        dt["~inv(H)"] = copy(state.invH)
        dt["Current step size"] = state.alpha
    end
    update!(
        tr,
        iteration,
        state.f_x,
        g_residual(state),
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
    )
end
