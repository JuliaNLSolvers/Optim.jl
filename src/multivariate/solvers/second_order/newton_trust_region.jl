#
# Check whether we are in the "hard case".
#
# Args:
#  H_eigv: The eigenvalues of H, low to high
#  qg: The inner products of the eigenvectors and the gradient, in the same order
#  ev_tol: Eigenvalues within ev_tol of H_eigv[1] belong to the bottom cluster.
#          Scale it to the Hessian (symmetric eigensolvers carry absolute error
#          on the order of eps times the spectral norm).
#  qg_tol: Gradient components below qg_tol count as zero: below the
#          gradient's own noise floor, or too small for the boundary root
#          they induce to be resolvable by the shifted solves.
#
# Returns:
#  hard_case: Whether it is a candidate for the hard case
#  lambda_index: The index of the first lambda not equal to the smallest
#                eigenvalue, which is only correct if hard_case is true.
function check_hard_case_candidate(H_eigv, qg, ev_tol, qg_tol)
    @assert length(H_eigv) == length(qg)
    if H_eigv[1] >= 0
        # The hard case is only when the smallest eigenvalue is negative.
        return false, 1
    end
    hard_case = true
    lambda_index = 1
    hard_case_check_done = false
    while !hard_case_check_done
        if lambda_index > length(H_eigv)
            hard_case_check_done = true
        elseif abs(H_eigv[1] - H_eigv[lambda_index]) > ev_tol
            # The eigenvalues are reported in order.
            hard_case_check_done = true
        else
            if abs(qg[lambda_index]) > qg_tol
                hard_case_check_done = true
                hard_case = false
            end
            lambda_index += 1
        end
    end

    hard_case, lambda_index
end

#==
The eigenbasis of H and the gradient expressed in it. The fields are named after
`eigvals` and `eigvecs`, which are what fill them.

This is the sub-problem's only input, and `refresh!` is the only thing that writes
it. Every write to g or H marks it out of date, by clearing `basis_current` on the
state, and the next solve rebuilds it; the two therefore cannot drift apart. A
rejected step restores the g and H this was built from and changes only delta, so
it leaves the flag set, which is what saves the O(n³) decomposition.

The rebuild is deferred to the solve rather than done at the write, so that a run
converging at its starting point never decomposes at all. That matters beyond the
work saved: `eigen` has no method for a `BigFloat` Hessian, and such a run is the
one case where `NewtonTrustRegion` still serves one (see issue #720).

A Hessian with no eigenbasis gives an all-NaN cache, which the solve screens for.
==#
struct TRSubproblemCache{Tg,TH}
    H_eigvals::Tg   # eigenvalues of H, ascending
    H_eigvecs::TH   # the matching eigenvectors, as columns
    qg::Tg          # H_eigvecs' * gr
end

# The `Symmetric` wrapper is what puts the eigenvalues in ascending order, which
# everything below relies on.
function refresh!(cache::TRSubproblemCache, gr::AbstractVector, H::AbstractMatrix)
    Hsym = Symmetric(H)
    if all(isfinite, Hsym)
        H_eig = eigen(Hsym)
        copyto!(cache.H_eigvals, H_eig.values)
        copyto!(cache.H_eigvecs, H_eig.vectors)
        mul!(cache.qg, cache.H_eigvecs', gr)
    else
        # `eigen` throws on a non-finite H, and a stale basis must not survive
        fill!(cache.H_eigvals, NaN)
        fill!(cache.H_eigvecs, NaN)
        fill!(cache.qg, NaN)
    end
    return cache
end

# Equation 4.38 in N&W (2006)
function calc_p!(lambda::T, min_i, n, spec::TRSubproblemCache, p) where {T}
    (; H_eigvals, H_eigvecs, qg) = spec
    fill!(p, zero(T))
    for i = min_i:n
        LinearAlgebra.axpy!(-qg[i] / (H_eigvals[i] + lambda), view(H_eigvecs, :, i), p)
    end
    return nothing
end

#==
The two squared norms that the Newton step of Algorithm 4.3 in N&W (2006) needs,
‖s(λ)‖² and ‖q(λ)‖² with q = R⁻ᵀs and RᵀR = H + λI, evaluated in the eigenbasis
of H rather than through a Cholesky factorization of H + λI.

Write H = Q*Diagonal(H_eigv)*Q' and qg = Q'gr. Equation 4.38 in that basis is
Q's = -qg./(H_eigv.+λ), and ‖q‖² = s'R⁻¹R⁻ᵀs = s'(H + λI)⁻¹s, so

    ‖s(λ)‖² = sum(qg[i]^2 / (H_eigv[i] + λ)^2)
    ‖q(λ)‖² = sum(qg[i]^2 / (H_eigv[i] + λ)^3)

Both cost O(n) and no factorization, which is what lets the root finder iterate
to convergence rather than stopping after a handful of decompositions.

`min_i` drops leading directions, as in `calc_p!`. Returns (‖s(λ)‖², ‖q(λ)‖²),
or (Inf, 0) when H + λI is not positive definite over the retained directions,
which tells the caller that λ is too small.
==#
function shifted_step_norms(
    H_eigv::AbstractVector{T},
    qg::AbstractVector{T},
    lambda::T,
    min_i::Int,
) where {T<:Real}
    n = length(qg)
    min_i <= n || return zero(T), zero(T)
    # The eigenvalues are ascending, so the smallest shifted one decides
    # definiteness for the whole range
    H_eigv[min_i] + lambda > 0 || return T(Inf), zero(T)

    norm2_s = zero(T)
    norm2_q = zero(T)
    # Summing from the back adds the smallest terms first, and dividing the
    # squared component once more avoids forming the cube of a small denominator
    for i = n:-1:min_i
        dλ = H_eigv[i] + lambda
        s2 = (qg[i] / dλ)^2
        norm2_s += s2
        norm2_q += s2 / dλ
    end
    return norm2_s, norm2_q
end

#==
The model value m(s) = gᵀs + sᵀHs/2 of the step that `calc_p!` forms for the same
lambda and min_i, summed over the same eigen-directions.

The step has components yᵢ = -qgᵢ/(Λᵢ+λ) for i ≥ min_i and zero below, and the
model separates over them:

    m = Σᵢ (qgᵢyᵢ + Λᵢyᵢ²/2) = -Σ_{i≥min_i} (qgᵢ/dᵢ)²(dᵢ + λ)/2,   dᵢ = Λᵢ + λ

Directions that min_i drops contribute exactly zero, because the step has no
component along them. So this is the value of the step actually returned, on every
branch, including the ones where (H + λI)s = -gr does not hold over the dropped
directions. Every term is non-positive for λ ≥ 0 and dᵢ > 0, so nothing cancels;
and unlike dot(s, H, s) none of them carries an absolute error of order
eps*‖H‖*‖s‖², which the model value itself can fall far below.
==#
function model_value_eigen(
    H_eigv::AbstractVector{T},
    qg::AbstractVector{T},
    lambda::T,
    min_i::Int,
) where {T<:Real}
    n = length(qg)
    m = zero(T)
    # Summed from the back, smallest terms first, as in `shifted_step_norms`
    for i = n:-1:min_i
        dλ = H_eigv[i] + lambda
        m -= (qg[i] / dλ)^2 * (dλ + lambda) / 2
    end
    return m
end

#==
A lambda strictly inside the bracket [lambda_lb, lambda_ub], both to start the
root finder and to retreat into the bracket after an overshoot. The geometric
mean is the Moré-Sorensen safeguard (p. 558 of [MORESORENSEN]); it collapses onto
a lower bound of zero, which is what the linear point covers. The square roots are
taken separately so a wide bracket cannot overflow the product.
==#
safeguarded_lambda(lambda_lb::T, lambda_ub::T) where {T<:Real} =
    max(lambda_lb + (lambda_ub - lambda_lb) / 100, sqrt(lambda_lb) * sqrt(lambda_ub))

# Choose a point in the trust region for the next step using
# the iterative (nearly exact) method of section 4.3 of N&W (2006).
# This is appropriate for Hessians that you factorize quickly.
#
# The sub-problem is stated entirely in the eigenbasis of H, so that is all this
# takes: `spec` holds the decomposition and the gradient expressed in it, and
# every quantity below is a sum over eigen-directions. Passing H and gr as well
# would be passing the same information twice, with nothing to keep the two
# copies in agreement.
#
# Args:
#  spec: The eigenbasis of H and the gradient in it. Read, never written.
#      See `TRSubproblemCache`.
#  delta:  The trust region size, ||s|| <= delta
#  s: Memory allocated for the step size, updated in place
#  tolerance: The relative tolerance on the boundary condition ‖s‖ == delta at
#      which root finding stops. `nothing` resolves to a few ulps, which is all
#      the eigendecomposition supports and which an O(n) root finding step can
#      afford. Pass a Real to override it.
#  max_iters: The maximum number of root finding iterations. Each iteration
#      excludes its own lambda from the bracket, so the iterates cannot cycle
#      and one of the two exits below is reached well inside the usual budget.
#      This only bounds the work in the worst case.
#
# Returns:
#  m - The numeric value of the quadratic minimization.
#  interior - A boolean indicating whether the solution was interior
#  lambda - The chosen regularizing quantity
#  hard_case - Whether or not it was a "hard case" as described by N&W (2006)
#  reached_solution - Whether the step returned solves the subproblem, judged on
#      the step itself rather than on which exit the root finder took
function solve_tr_subproblem!(
    spec::TRSubproblemCache,
    delta::T,
    s::AbstractVector{T};
    tolerance::Union{Real,Nothing} = nothing,
    max_iters::Int = 100,
) where {T<:Real}
    (; H_eigvals, H_eigvecs, qg) = spec
    n = length(qg)
    delta_sq = delta^2

    @assert n == length(s)
    @assert max_iters >= 1

    # Q is orthogonal, so the gradient keeps its norm in the eigenbasis
    gr_norm = norm(qg)

    # A Hessian with no eigenbasis leaves the cache NaN, and there is no
    # subproblem to solve in that basis. A non-finite gradient reaches here the
    # same way, through qg.
    if iszero(n) || !isfinite(H_eigvals[1]) || !isfinite(gr_norm)
        # Leave a well-defined (zero) step behind: callers read s after this
        # returns, and a stale or NaN-filled s poisons the radius update.
        fill!(s, zero(T))
        return T(Inf), false, zero(T), false, false
    end

    min_H_ev, max_H_ev = H_eigvals[1], H_eigvals[n]
    H_scale = max(abs(min_H_ev), abs(max_H_ev)) # spectral norm

    if iszero(H_scale)
        # H == 0: the model is linear, and every tolerance below degenerates.
        # The minimizer over the ball is the boundary Cauchy point, or the
        # origin for a zero gradient; both are exact.
        if iszero(gr_norm)
            fill!(s, zero(T))
            return zero(T), true, zero(T), false, true
        else
            # gr = Q*qg, so the Cauchy point costs one matrix-vector product
            mul!(s, H_eigvecs, qg)
            rmul!(s, -delta / gr_norm)
            return -delta * gr_norm, false, gr_norm / delta, false, true
        end
    end

    # All classification thresholds are relative and typed. Eigenvalue-space
    # quantities compare against the spectral norm; gradient-space quantities
    # (the qg components) compare against the gradient norm.
    scale_tol = sqrt(eps(T)) * H_scale
    gr_tol = sqrt(eps(T)) * gr_norm

    # These values describe the outcome of the subproblem.  They will be
    # set below and returned at the end.
    interior = true
    hard_case = false
    reached_solution = true

    # Unconstrained solution. The gate is a relative condition test: the
    # historical absolute 1e-8 misread scaled Hessians in both directions.
    positive_definite = min_H_ev > scale_tol
    if positive_definite
        calc_p!(zero(T), 1, n, spec, s)
    end

    if positive_definite && sum(abs2, s) <= delta_sq
        # No shrinkage is necessary: -(H \ gr) is the minimizer
        interior = true
        reached_solution = true
        lambda = zero(T)
        m = model_value_eigen(H_eigvals, qg, lambda, 1)
    else
        interior = false

        # The hard case is when the gradient is orthogonal to all
        # eigenvectors associated with the lowest eigenvalue.
        # A gradient component along the bottom cluster counts as zero for
        # hard-case candidacy when it is below the gradient's noise floor, or
        # when it is too small for the boundary root it induces to be
        # resolvable: that root sits within qg/delta of -min_H_ev, which the
        # eigensolver itself pins down only to sqrt(eps)*H_scale. The
        # conditioning floor applies only here, where min_H_ev < 0; the
        # classification below judges a component against gr_tol alone, so
        # resolvable ones are not dropped from the step.
        qg_tol = max(gr_tol, scale_tol * delta)
        hard_case_candidate, min_i =
            check_hard_case_candidate(H_eigvals, qg, scale_tol, qg_tol)

        # The multiplier is bounded below by feasibility (lambda >= 0) and by
        # positive semidefiniteness of H + lambda*I, and above by the root of
        # the norm bound ‖s(lambda)‖ <= ‖g‖/(min_H_ev + lambda) (Geyer's
        # lambda_up; also p. 558 of [MORESORENSEN]).
        lambda_lb = max(zero(T), nextfloat(-min_H_ev))
        lambda_ub = max(gr_norm / delta - min_H_ev, lambda_lb)
        lambda = lambda_lb

        # Root finding stops on the boundary condition ‖s(lambda)‖ == delta,
        # relative to delta. The sum forming ‖s(lambda)‖² carries about one ulp
        # per term, so n ulps is the accuracy the computation has to give.
        phi_tol = tolerance === nothing ? n * eps(T) : T(tolerance)

        hard_case = false
        p_lambda2 = zero(T) # ‖p(lambda_lb)‖², meaningful only in the hard case
        if hard_case_candidate
            # The "hard case". lambda is taken to be -min_H_ev and we only need
            # to find a multiple of an orthogonal eigenvector that lands the
            # iterate on the boundary.

            # Formula 4.45 in N&W (2006). The step it gives settles the case: it
            # is the hard case only if that step fits inside the region.
            calc_p!(lambda, min_i, n, spec, s)
            p_lambda2 = sum(abs2, s)
            hard_case = p_lambda2 <= delta_sq
        end

        # ‖s(lambda)‖ decreases in lambda, so over the feasible range it is
        # largest at lambda_lb. If even that step lies inside the region there is
        # no boundary solution to find, and the minimizer is the interior step at
        # lambda_lb. A direction with H_eigvals[i] + lambda_lb ≈ 0 sends
        # ‖s‖ to infinity unless its gradient component vanishes, in which case
        # it drops out of the sum and out of the step. "Vanishes" is judged on
        # the gradient's own scale: comparing qg against an H-scaled tolerance
        # drops components whose qg/d ratio is large, which truncates the step.
        # The eigenvalue-space tolerance here must equal the cluster tolerance
        # in check_hard_case_candidate: both tests examine the same quantity,
        # and unequal tolerances give contradictory classifications.
        norm2_lb = zero(T)
        first_nz = 1
        boundary_solution_exists = false
        for i = 1:n
            d = H_eigvals[i] + lambda_lb
            if d > scale_tol
                norm2_lb += (qg[i] / d)^2
            elseif abs(qg[i]) > gr_tol
                boundary_solution_exists = true
            elseif first_nz == i
                # Only a leading run of negligible components drops out of the
                # step, so a significant one is never skipped over
                first_nz = i + 1
            end
        end
        interior_at_lb = !boundary_solution_exists && norm2_lb <= delta_sq

        # One arm per outcome, each forming the step and the model value of that
        # same step. Written as a single chain so every path through it assigns
        # both, rather than leaving that to an invariant across branches.
        if hard_case
            reached_solution = true

            # Either sign lands on the boundary, and in the hard case proper the
            # two give the same model value, since the gradient has no component
            # along this eigenvector. It has a small one whenever the screen only
            # judged it negligible, and then this sign is the one that spends it
            # on a decrease.
            tau = -copysign(sqrt(delta_sq - p_lambda2), qg[1])

            # Formula 4.45 is s = p + tau*z where z is any unit eigenvector for
            # the smallest eigenvalue; s already holds p, so add tau times the
            # first eigenvector.
            LinearAlgebra.axpy!(tau, view(H_eigvecs, :, 1), s)

            # p contributes over the retained directions, and the tau term over
            # the first one, which p leaves empty
            m =
                model_value_eigen(H_eigvals, qg, lambda, min_i) +
                (qg[1] + H_eigvals[1] * tau / 2) * tau
        elseif interior_at_lb
            calc_p!(lambda_lb, first_nz, n, spec, s)
            interior = true
            reached_solution = true
            lambda = lambda_lb
            m = model_value_eigen(H_eigvals, qg, lambda, first_nz)
        else
            # Algorithm 4.3 of N&W (2006), with s instead of p_l for consistency
            # with Optim.jl. Both norms come from `shifted_step_norms`, so an
            # iteration is O(n) and needs no factorization of H + lambda*I.
            #
            # Start strictly inside the bracket, at the same safeguarded point
            # the loop retreats to. The Moré-Sorensen starting bounds (p. 560)
            # are built from ‖H‖₁ and the diagonal because a factorization-based
            # solver has no spectrum to consult; here the bracket already holds
            # the exact eigenvalue bounds, and it is tighter than theirs.
            lambda = safeguarded_lambda(lambda_lb, lambda_ub)

            for iter = 1:max_iters
                lambda_previous = lambda
                norm2_s, norm2_q = shifted_step_norms(H_eigvals, qg, lambda, first_nz)

                # ‖s(lambda)‖ decreases in lambda, so the sign of phi places
                # lambda relative to the root and tightens the bracket
                phi = sqrt(norm2_s) - delta
                abs(phi) <= phi_tol * delta && break
                if phi > 0
                    lambda_lb = lambda
                else
                    lambda_ub = lambda
                end

                lambda += norm2_s * phi / (delta * norm2_q)
                if !(lambda_lb < lambda < lambda_ub)
                    # An overshoot, or a lambda too small to keep H + lambda*I
                    # positive definite; retreat into the bracket
                    lambda = safeguarded_lambda(lambda_lb, lambda_ub)
                end

                # The iterates have stopped moving on their own scale
                abs(lambda - lambda_previous) <= 4 * eps(T) * lambda && break
            end

            # Forming the step from the returned lambda keeps the two describing
            # the same solution of (H + lambda*I)s = -gr
            calc_p!(lambda, first_nz, n, spec, s)
            reached_solution = abs(norm(s) - delta) <= cbrt(eps(T)) * delta
            m = model_value_eigen(H_eigvals, qg, lambda, first_nz)
        end
    end

    return m, interior, lambda, hard_case, reached_solution
end

# The same sub-problem for a caller holding H and gr rather than their
# decomposition. It decomposes them and forwards; `NewtonTrustRegion` keeps the
# decomposition on its state instead, so that a rejected step, which changes only
# delta, does not repeat it.
function solve_tr_subproblem!(
    gr::AbstractVector{T},
    H::AbstractMatrix{T},
    delta::T,
    s::AbstractVector{T};
    tolerance::Union{Real,Nothing} = nothing,
    max_iters::Int = 100,
) where {T<:Real}
    spec = refresh!(TRSubproblemCache(similar(gr), similar(H), similar(gr)), gr, H)
    return solve_tr_subproblem!(spec, delta, s; tolerance, max_iters)
end

struct NewtonTrustRegion{T<:Real} <: SecondOrderOptimizer
    initial_delta::T
    delta_hat::T
    delta_min::T
    eta::T
    rho_lower::T
    rho_upper::T
    use_fg::Bool

    function NewtonTrustRegion(
        initial_delta::T,
        delta_hat::T,
        delta_min::T,
        eta::T,
        rho_lower::T,
        rho_upper::T,
        use_fg::Bool,
    ) where {T<:Real}
        if !(delta_hat > 0)
            throw(DomainError(delta_hat, "maximum trust region radius must be positive"))
        end
        if !(0 < initial_delta < delta_hat)
            throw(DomainError(initial_delta, LazyString("initial trust region radius must be positive and below the maximum trust region radius (", delta_hat, ")")))
        end
        if !(delta_min >= 0)
            throw(DomainError(delta_min, "smallest allowable trust region radius must be non-negative"))
        end
        if !(eta >= 0)
            throw(DomainError(eta, "minimum threshold of actual and predicted reduction for accepting a step must be non-negative"))
        end
        if !(rho_lower > eta)
            throw(DomainError(rho_lower, LazyString("maximum threshold of actual and predicted reduction for shrinking the trust region must be greater than the minimum threshold for accepting a step (", eta, ")")))
        end
        if !(rho_upper > rho_lower)
            throw(DomainError(rho_upper, LazyString("minimum threshold of actual and predicted reduction for growing the trust region must be greater than the minimum threshold for shrinking it (", rho_lower, ")")))
        end

        return new{T}(initial_delta, delta_hat, delta_min, eta, rho_lower, rho_upper, use_fg)
    end
end

"""
# NewtonTrustRegion
## Constructor
```julia
NewtonTrustRegion(; initial_delta = 1.0,
                    delta_hat = 100.0,
                    delta_min = 0.0,
                    eta = 0.1,
                    rho_lower = 0.25,
                    rho_upper = 0.75,
                    use_fg = true)
```

The constructor has 7 keywords:
* `initial_delta`, the initial trust region radius. Defaults to `1.0`.
* `delta_hat`, the largest allowable trust region radius. Defaults to `100.0`.
* `delta_min`, the smallest allowable trust region radius. Optimization halts if the updated radius is less than or equal to this value. Defaults to `0.0`.
* `eta`, when the ratio of actual and predicted reduction is greater than `eta`, accept the step. Defaults to `0.1`.
* `rho_lower`, when the ratio of actual and predicted reduction is less than `rho_lower`, shrink the trust region. Defaults to `0.25`.
* `rho_upper`, when the ratio of actual and predicted reduction is greater than `rho_upper` and the proposed step is at the boundary of the trust region, grow the trust region. Defaults to `0.75`.
* `use_fg`, when true always evaluate the gradient with the value after solving the subproblem. This is more efficient if f and g share expensive computations. Defaults to `true`.

## Description
The `NewtonTrustRegion` method implements Newton's method with a trust region
for optimizing a function. The method is designed to take advantage of the
second-order information in a function's Hessian, but with more stability that
Newton's method when functions are not globally well-approximated by a quadratic.
This is achieved by repeatedly minimizing quadratic approximations within a
dynamically-sized trust region in which the function is assumed to be locally
quadratic. See Wright and Nocedal and Wright (ch. 4, 2006) for a discussion of
trust-region methods in practice.

## References
 - Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer Science & Business Media.
"""
function NewtonTrustRegion(;
    initial_delta::Real = 1.0,
    delta_hat::Real = 100.0,
    delta_min::Real = 0.0,
    eta::Real = 0.1,
    rho_lower::Real = 0.25,
    rho_upper::Real = 0.75,
    use_fg::Bool = true,
)
    NewtonTrustRegion(promote(initial_delta, delta_hat, delta_min, eta, rho_lower, rho_upper)..., use_fg)
end

Base.summary(io::IO, ::NewtonTrustRegion) = print(io, "Newton's Method (Trust Region)")

mutable struct NewtonTrustRegionState{Tx,T,Tg,TH,TC<:TRSubproblemCache} <:
               AbstractOptimizerState
    const x::Tx
    const g_x::Tg
    const H_x::TH
    f_x::T
    const x_previous::Tx
    f_x_previous::T
    const s::Tx
    const x_cache::Tx
    const g_cache::Tg
    hard_case::Bool
    reached_subproblem_solution::Bool
    interior::Bool
    delta::T
    lambda::T
    const eta::T
    rho::T
    const subproblem_cache::TC
    basis_current::Bool
end

# NaN is false here, which is what rejects a nonfinite trial point
step_accepted(state::NewtonTrustRegionState) = state.rho > state.eta

function initial_state(method::NewtonTrustRegion, options, d, x0)
    T = eltype(x0)
    # Keep track of trust region sizes
    delta = copy(method.initial_delta)

    # Record attributes of the subproblem in the trace.
    hard_case = false
    reached_subproblem_solution = true
    interior = true
    lambda = NaN

    f_x, g_x, H_x = NLSolversBase.value_gradient_hessian!(d, x0)

    NewtonTrustRegionState(
        copy(x0), # Maintain current state in state.x
        copy(g_x), # Maintain current gradient in state.g_x
        copy(H_x), # Maintain current Hessian in state.H_x
        f_x, # Maintain current f in state.f_x
        fill!(similar(x0), NaN), # Maintain previous state in state.x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        fill!(similar(x0), NaN), # Maintain current search direction in state.s
        fill!(similar(x0), NaN), # Cache to be able to reset state.x
        fill!(method.use_fg ? similar(g_x) : empty(g_x), NaN), # Cache to be able to reset state.g_x
        hard_case,
        reached_subproblem_solution,
        interior,
        T(delta),
        T(lambda),
        T(method.eta), # eta
        zero(T), # rho
        TRSubproblemCache(
            fill!(similar(g_x), NaN),
            fill!(similar(H_x), NaN),
            fill!(similar(g_x), NaN),
        ),
        false, # basis_current: g and H are new, so the first solve decomposes
    )
end


function update_state!(d::TwiceDifferentiable, state::NewtonTrustRegionState, method::NewtonTrustRegion)
    # Rebuild the basis if g or H have been written since it was built. A
    # rejected step writes neither, and shrinking delta does not move it.
    if !state.basis_current
        refresh!(state.subproblem_cache, state.g_x, state.H_x)
        state.basis_current = true
    end

    # Find the next step direction
    m, state.interior, state.lambda, state.hard_case, state.reached_subproblem_solution =
        solve_tr_subproblem!(state.subproblem_cache, state.delta, state.s)

    # Maintain a record of current position, to be able to reset it below
    copyto!(state.x_cache, state.x)
    f_cache = state.f_x

    # Update current position
    state.x .+= state.s

    # Update the function value and gradient
    if method.use_fg
        copyto!(state.g_cache, state.g_x)
        f_x, g_x = value_gradient!(d, state.x)
        copyto!(state.g_x, g_x)
        state.f_x = f_x
    else
        f_x = value!(d, state.x)
        state.f_x = f_x
    end
    # Update the trust region size based on the discrepancy between
    # the predicted and actual function values.  (Algorithm 4.1 in N&W (2006))
    f_x_diff = f_cache - f_x
    if abs(m) <= eps(typeof(m))
        # This should only happen when the step is very small, in which case
        # we should accept the step and assess_convergence(). There is no
        # predicted reduction to compare against, so the only thing left to
        # check is that the objective did not actually go up.
        state.rho = f_x_diff >= 0 ? one(state.rho) : -one(state.rho)
    elseif m > 0
        # This can happen if the trust region radius is too large and the
        # Hessian is not positive definite.  We should shrink the trust
        # region.
        state.rho = -one(state.rho)
    else
        state.rho = f_x_diff / (- m)
    end

    # The step is accepted if the ratio is greater than eta
    accept_step = step_accepted(state)

    # Update trust region radius
    if !accept_step
        # The improvement is too small and we won't take it.
        # If you reject an interior solution, make sure that the next
        # delta is smaller than the current step (state.s). Otherwise you waste
        # steps reducing delta by constant factors while each solution
        # will be the same. If this keeps on happening it could be a sign
        # errors in the gradient or a non-differentiability at the optimum.
        # A rejection must never enlarge the radius: an unconverged subproblem
        # can return ‖s‖ far above delta (and a non-finite H can make it NaN),
        # and norm(s)/4 alone would then blow the radius up instead of
        # shrinking it, so cap by the current delta.
        s_norm = norm(state.s)
        state.delta = (isfinite(s_norm) ? min(state.delta, s_norm) : state.delta) / 4
    elseif state.rho < method.rho_lower
        state.delta /= 4
    elseif (state.rho > method.rho_upper) && !state.interior
        state.delta = min(2 * state.delta, method.delta_hat)
    end

    # Update/reset gradients and function values
    if accept_step
        if method.use_fg
            copyto!(state.H_x, hessian!(d, state.x))
        else
            g_x, H_x = NLSolversBase.gradient_hessian!(d, state.x)
            copyto!(state.g_x, g_x)
            copyto!(state.H_x, H_x)
        end
        # g and H have just been replaced, so the basis no longer describes them
        state.basis_current = false
        # Update history
        copyto!(state.x_previous, state.x_cache)
        state.f_x_previous = f_cache
    else
        # Reset state
        copyto!(state.x, state.x_cache)
        state.f_x = f_cache
        if method.use_fg
            copyto!(state.g_x, state.g_cache)
        end
    end

    false
end

function assess_convergence(state::NewtonTrustRegionState, d, options::Options)
    if step_accepted(state)
        # Accept the point and check convergence against all five tolerances,
        # like the default path used by the first-order solvers. The 8-argument
        # method only honors x_abstol and f_reltol, silently ignoring x_reltol
        # and f_abstol.
        return assess_convergence(
            state.x,
            state.x_previous,
            state.f_x,
            state.f_x_previous,
            state.g_x,
            options.x_abstol,
            options.x_reltol,
            options.f_abstol,
            options.f_reltol,
            options.g_abstol,
        )
    else
        return false, false, false, false
    end
end

# Function value, gradient and Hessian matrix are already updated in update_state!
update_fgh!(d, state, ::NewtonTrustRegion) = nothing

function trace!(
    tr,
    d,
    state::NewtonTrustRegionState,
    iteration::Integer,
    ::NewtonTrustRegion,
    options::Options,
    curr_time = time(),
)
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g_x)
        dt["h(x)"] = copy(state.H_x)
        dt["delta"] = copy(state.delta)
        dt["interior"] = state.interior
        dt["hard case"] = state.hard_case
        dt["reached_subproblem_solution"] = state.reached_subproblem_solution
        dt["lambda"] = state.lambda
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
