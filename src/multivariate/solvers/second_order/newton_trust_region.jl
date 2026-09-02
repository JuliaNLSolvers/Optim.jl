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
#          they induce to be resolvable by the ridged solves.
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

# Equation 4.38 in N&W (2006)
function calc_p!(lambda::T, min_i, n, qg, H_eig, p) where {T}
    fill!(p, zero(T))
    for i = min_i:n
        LinearAlgebra.axpy!(-qg[i] / (H_eig.values[i] + lambda), view(H_eig.vectors, :, i), p)
    end
    return nothing
end

#==
Returns a tuple of initial safeguarding values for λ. Newton's method might not
work well without these safeguards when the Hessian is not positive definite.
==#
function initial_safeguards(H, gr, delta, lambda)
    # equations are on p. 560 of [MORESORENSEN]
    T = eltype(gr)
    λS = -Base.minimum(@view(H[diagind(H)])) # Base.minimum !== minimum
    # they state on the first page that ||⋅|| is the Euclidean norm
    gr_norm = norm(gr)
    Hnorm = opnorm(H, 1)
    λL = max(T(0), λS, gr_norm / delta - Hnorm)
    λU = gr_norm / delta + Hnorm
    # p. 558
    lambda = clamp(lambda, λL, λU)
    if lambda ≤ λS
        lambda = max(T(1) / 1000 * λU, sqrt(λL * λU))
    end
    lambda
end

# Choose a point in the trust region for the next step using
# the iterative (nearly exact) method of section 4.3 of N&W (2006).
# This is appropriate for Hessians that you factorize quickly.
#
# Args:
#  gr: The gradient
#  H:  The Hessian
#  delta:  The trust region size, ||s|| <= delta
#  s: Memory allocated for the step size, updated in place
#  tolerance: The convergence tolerance for root finding. The default
#      (`nothing`) resolves to eps(T)^(2/3) times the width of the bracket
#      containing the boundary root, which reproduces the historical absolute
#      1e-10 at unit scale in Float64; pass a Real to override it.
#  max_iters: The maximum number of root finding iterations
#
# Returns:
#  m - The numeric value of the quadratic minimization.
#  interior - A boolean indicating whether the solution was interior
#  lambda - The chosen regularizing quantity
#  hard_case - Whether or not it was a "hard case" as described by N&W (2006)
#  reached_solution - Whether or not a solution was reached (as opposed to
#      terminating early due to max_iters)
function solve_tr_subproblem!(gr, H, delta, s; tolerance = nothing, max_iters = 5)
    T = eltype(gr)
    n = length(gr)
    delta_sq = delta^2

    @assert n == length(s)
    @assert (n, n) == size(H)
    @assert max_iters >= 1

    # Note that currently the eigenvalues are only sorted if H is perfectly
    # symmetric.  (Julia issue #17093)
    Hsym = Symmetric(H)
    if any(!isfinite, Hsym)
        # Leave a well-defined (zero) step behind: callers read s after this
        # returns, and a stale or NaN-filled s poisons the radius update.
        fill!(s, zero(T))
        return T(Inf), false, zero(T), false, false
    end
    H_eig = eigen(Hsym)

    if !isempty(H_eig.values)
        min_H_ev, max_H_ev = H_eig.values[1], H_eig.values[n]
    else
        fill!(s, zero(T))
        return T(Inf), false, zero(T), false, false
    end
    H_scale = max(abs(min_H_ev), abs(max_H_ev)) # spectral norm
    gr_norm = norm(gr)

    if iszero(H_scale)
        # H == 0: the model is linear, and every tolerance below degenerates.
        # The minimizer over the ball is the boundary Cauchy point, or the
        # origin for a zero gradient; both are exact.
        if iszero(gr_norm)
            fill!(s, zero(T))
            return zero(T), true, zero(T), false, true
        else
            s .= (-delta / gr_norm) .* gr
            return -delta * gr_norm, false, gr_norm / delta, false, true
        end
    end

    # All classification thresholds are relative and typed. Eigenvalue-space
    # quantities compare against the spectral norm; gradient-space quantities
    # (the qg components) compare against the gradient norm.
    scale_tol = sqrt(eps(T)) * H_scale
    gr_tol = sqrt(eps(T)) * gr_norm

    H_ridged = copy(H)

    # Cache the inner products between the eigenvectors and the gradient.
    qg = H_eig.vectors' * gr

    # These values describe the outcome of the subproblem.  They will be
    # set below and returned at the end.
    interior = true
    hard_case = false
    reached_solution = true

    # Unconstrained solution. The gate is a relative condition test: the
    # historical absolute 1e-8 misread scaled Hessians in both directions.
    positive_definite = min_H_ev > scale_tol
    if positive_definite
        calc_p!(zero(T), 1, n, qg, H_eig, s)
    end

    if positive_definite && sum(abs2, s) <= delta_sq
        # No shrinkage is necessary: -(H \ gr) is the minimizer
        interior = true
        reached_solution = true
        lambda = zero(T)
    else
        interior = false

        # The hard case is when the gradient is orthogonal to all
        # eigenvectors associated with the lowest eigenvalue.
        # A gradient component along the bottom cluster counts as zero for
        # hard-case candidacy when it is below the gradient's noise floor, or
        # when it is too small for the boundary root it induces to be
        # resolvable: that root sits within qg/delta of -min_H_ev, and the
        # ridged solves cannot resolve offsets below sqrt(eps)*H_scale. The
        # conditioning floor applies only here, where min_H_ev < 0; the
        # interior classification below computes its step in the eigenbasis,
        # where such components are resolvable and must not be dropped.
        qg_tol = max(gr_tol, scale_tol * delta)
        hard_case_candidate, min_i =
            check_hard_case_candidate(H_eig.values, qg, scale_tol, qg_tol)

        # The multiplier is bounded below by feasibility (lambda >= 0) and by
        # positive semidefiniteness of H + lambda*I, and above by the root of
        # the norm bound ‖s(lambda)‖ <= ‖g‖/(min_H_ev + lambda) (Geyer's
        # lambda_up; also p. 558 of [MORESORENSEN]).
        lambda_lb = max(zero(T), nextfloat(-min_H_ev))
        lambda_ub = gr_norm / delta - min_H_ev
        lambda = lambda_lb

        # The boundary root lives inside [lambda_lb, lambda_ub], an interval of
        # width at most ‖g‖/delta, so the increment tolerance scales with the
        # bracket width, floored by a few ulps of the iterates' own magnitude.
        # eps^(2/3) reproduces the historical absolute 1e-10 at unit scale.
        lambda_tol =
            tolerance === nothing ?
            max(
                cbrt(eps(T))^2 * (lambda_ub - lambda_lb),
                4 * eps(T) * lambda_ub,
            ) : T(tolerance)

        hard_case = false
        if hard_case_candidate
            # The "hard case". lambda is taken to be -min_H_ev and we only need
            # to find a multiple of an orthogonal eigenvector that lands the
            # iterate on the boundary.

            # Formula 4.45 in N&W (2006)
            calc_p!(lambda, min_i, n, qg, H_eig, s)
            p_lambda2 = sum(abs2, s)
            if p_lambda2 > delta_sq
                # Then we can simply solve using root finding.
            else
                hard_case = true
                reached_solution = true

                tau = sqrt(delta_sq - p_lambda2)

                # Formula 4.45 is s = p + tau*z where z is any unit eigenvector
                # for the smallest eigenvalue; s already holds p, so add tau
                # times the first eigenvector.
                LinearAlgebra.axpy!(tau, view(H_eig.vectors, :, 1), s)
            end
        end

        # ‖s(lambda)‖ decreases in lambda, so over the feasible range it is
        # largest at lambda_lb. If even that step lies inside the region there is
        # no boundary solution to find, and the minimizer is the interior step at
        # lambda_lb. A direction with H_eig.values[i] + lambda_lb ≈ 0 sends
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
            d = H_eig.values[i] + lambda_lb
            if d <= scale_tol
                first_nz = i + 1
                if abs(qg[i]) > gr_tol
                    boundary_solution_exists = true
                    break
                end
            else
                norm2_lb += (qg[i] / d)^2
            end
        end
        interior_at_lb = !boundary_solution_exists && norm2_lb <= delta_sq

        if !hard_case && interior_at_lb
            calc_p!(lambda_lb, first_nz, n, qg, H_eig, s)
            interior = true
            reached_solution = true
            lambda = lambda_lb
        elseif !hard_case
            lambda = initial_safeguards(H, gr, delta, lambda)
            # Algorithm 4.3 of N&W (2006), with s instead of p_l for consistency
            # with Optim.jl

            reached_solution = false
            # Factorization failures draw on their own budget so they do not
            # consume root-finding iterations: with max_iters = 5, a single
            # failed Cholesky otherwise turns a run that converges in 5 root
            # steps into reached_solution = false. Failures occur only while
            # lambda <= -min_H_ev <= H_scale, and each failure at least doubles
            # lambda starting from sqrt(eps(T)) * H_scale, so the number of
            # failures is bounded by the doublings needed to cross H_scale,
            # about half the significand width.
            max_retries = 4 + ceil(Int, precision(T) / 2)
            retries = 0
            iter = 0
            while iter < max_iters
                lambda_previous = lambda

                for i in diagind(H_ridged)
                    H_ridged[i] = H[i] + lambda
                end

                F = cholesky(Hermitian(H_ridged), check = false)
                # Sometimes, λ is not sufficiently large for the Cholesky factorization
                # to succeed. In that case, we increase λ and retry.
                # Merely doubling λ is not generally sufficient to make H + λI numerically
                # positive-definite: e.g., if λ ~ 1e-15, we would never reach a stable
                # regime, which would leave `s` unchanged. Instead, jump
                # to a ridge on the order of H's spectral scale so the next factorization
                # succeeds; the root-finder can still descend toward a smaller optimal λ
                # afterwards, since `lambda_lb` is left at its initial value
                if !issuccess(F)
                    retries += 1
                    retries > max_retries && break
                    lambda = max(2 * lambda, sqrt(eps(T)) * H_scale)
                    continue
                end
                iter += 1

                R = F.U
                s[:] = -R \ (R' \ gr)
                q_l = R' \ s
                norm2_s = dot(s, s)
                lambda_update = norm2_s * (sqrt(norm2_s) - delta) / (delta * dot(q_l, q_l))
                lambda += lambda_update

                # Keep lambda inside the bracket [lambda_lb, lambda_ub]: a
                # boundary root, when it exists, lies in it, so an iterate
                # outside is an overshoot; go half the way back to the bound.
                if lambda < lambda_lb
                    lambda = (lambda_previous + lambda_lb) / 2
                elseif lambda > lambda_ub
                    lambda = (lambda_previous + lambda_ub) / 2
                end

                if abs(lambda - lambda_previous) < lambda_tol
                    # The lambda iterates have stopped moving. That means the
                    # boundary root was found only if the step in hand actually
                    # sits on the boundary; the same test also triggers on
                    # safeguard stagnation at a bound, where the step does not.
                    reached_solution = abs(sqrt(norm2_s) - delta) <= cbrt(eps(T)) * delta
                    break
                end
            end
        end
    end

    m = dot(gr, s) + dot(s, H, s) / 2

    return m, interior, lambda, hard_case, reached_solution
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
            throw(DomainError(initial_delta, LazyString("initial trust region radius must be positive and below the maiximum trust region radius (", delta_hat, ")")))
        end
        if !(delta_min >= 0)
            throw(DomainError(delta_min, "smallest allowable trust region radius must be non-negative"))
        end
        if !(eta >= 0)
            throw(DomainError(eta, "minimum threshold of actual and predicted reduction for accepting a step must be positivethreshold eta must be non-negative"))
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

mutable struct NewtonTrustRegionState{Tx,T,Tg,TH} <: AbstractOptimizerState
    x::Tx
    g_x::Tg
    H_x::TH
    f_x::T
    x_previous::Tx
    f_x_previous::T
    s::Tx
    x_cache::Tx
    g_cache::Tg
    hard_case::Bool
    reached_subproblem_solution::Bool
    interior::Bool
    delta::T
    lambda::T
    eta::T
    rho::T
end

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
        zero(T),
    ) # rho
end


function update_state!(d::TwiceDifferentiable, state::NewtonTrustRegionState, method::NewtonTrustRegion)
    # Find the next step direction.
    m, state.interior, state.lambda, state.hard_case, state.reached_subproblem_solution =
        solve_tr_subproblem!(state.g_x, state.H_x, state.delta, state.s)

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
        # we should accept the step and assess_convergence().
        state.rho = 1.0
    elseif m > 0
        # This can happen if the trust region radius is too large and the
        # Hessian is not positive definite.  We should shrink the trust
        # region.
        state.rho = -1.0
    else
        state.rho = f_x_diff / (- m)
    end

    # The step is accepted if the ratio is greater than eta
    accept_step = state.rho > state.eta

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
    if state.rho > state.eta
        # Accept the point and check convergence
        return assess_convergence(
            state.x,
            state.x_previous,
            state.f_x,
            state.f_x_previous,
            state.g_x,
            options.x_abstol,
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
