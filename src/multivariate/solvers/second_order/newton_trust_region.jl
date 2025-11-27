#
# Check whether we are in the "hard case".
#
# Args:
#  H_eigv: The eigenvalues of H, low to high
#  qg: The inner product of the eigenvalues and the gradient in the same order
#
# Returns:
#  hard_case: Whether it is a candidate for the hard case
#  lambda_index: The index of the first lambda not equal to the smallest
#                eigenvalue, which is only correct if hard_case is true.
function check_hard_case_candidate(H_eigv, qg)
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
        elseif abs(H_eigv[1] - H_eigv[lambda_index]) > 1e-10
            # The eigenvalues are reported in order.
            hard_case_check_done = true
        else
            if abs(qg[lambda_index]) > 1e-10
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
# the interative (nearly exact) method of section 4.3 of N&W (2006).
# This is appropriate for Hessians that you factorize quickly.
#
# Args:
#  gr: The gradient
#  H:  The Hessian
#  delta:  The trust region size, ||s|| <= delta
#  s: Memory allocated for the step size, updated in place
#  tolerance: The convergence tolerance for root finding
#  max_iters: The maximum number of root finding iterations
#
# Returns:
#  m - The numeric value of the quadratic minimization.
#  interior - A boolean indicating whether the solution was interior
#  lambda - The chosen regularizing quantity
#  hard_case - Whether or not it was a "hard case" as described by N&W (2006)
#  reached_solution - Whether or not a solution was reached (as opposed to
#      terminating early due to max_iters)
function solve_tr_subproblem!(gr, H, delta, s; tolerance = 1e-10, max_iters = 5)
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
        return T(Inf), false, zero(T), false, false
    end
    H_eig = eigen(Hsym)

    if !isempty(H_eig.values)
        min_H_ev, max_H_ev = H_eig.values[1], H_eig.values[n]
    else
        return T(Inf), false, zero(T), false, false
    end
    H_ridged = copy(H)

    # Cache the inner products between the eigenvectors and the gradient.
    qg = H_eig.vectors' * gr

    # These values describe the outcome of the subproblem.  They will be
    # set below and returned at the end.
    interior = true
    hard_case = false
    reached_solution = true

    # Unconstrained solution
    if min_H_ev >= 1e-8
        calc_p!(zero(T), 1, n, qg, H_eig, s)
    end

    if min_H_ev >= 1e-8 && sum(abs2, s) <= delta_sq
        # No shrinkage is necessary: -(H \ gr) is the minimizer
        interior = true
        reached_solution = true
        lambda = zero(T)
    else
        interior = false

        # The hard case is when the gradient is orthogonal to all
        # eigenvectors associated with the lowest eigenvalue.
        hard_case_candidate, min_i = check_hard_case_candidate(H_eig.values, qg)

        # Solutions smaller than this lower bound on lambda are not allowed:
        # they don't ridge H enough to make H_ridge PSD.
        lambda_lb = nextfloat(-min_H_ev)
        lambda = lambda_lb

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

                # I don't think it matters which eigenvector we pick so take
                # the first.
                calc_p!(lambda, min_i, n, qg, H_eig, s)
                LinearAlgebra.axpby!(tau, view(H_eig.vectors, :, 1), -1, s)
            end
        end

        lambda = initial_safeguards(H, gr, delta, lambda)

        if !hard_case
            # Algorithim 4.3 of N&W (2006), with s insted of p_l for consistency
            # with Optim.jl

            reached_solution = false
            for iter = 1:max_iters
                lambda_previous = lambda

                for i in diagind(H_ridged)
                    H_ridged[i] = H[i] + lambda
                end

                F = cholesky(Hermitian(H_ridged), check = false)
                # Sometimes, lambda is not sufficiently large for the Cholesky factorization
                # to succeed. In that case, we set double lambda and continue to next iteration
                if !issuccess(F)
                    lambda *= 2
                    continue
                end

                R = F.U
                s[:] = -R \ (R' \ gr)
                q_l = R' \ s
                norm2_s = dot(s, s)
                lambda_update = norm2_s * (sqrt(norm2_s) - delta) / (delta * dot(q_l, q_l))
                lambda += lambda_update

                # Check that lambda is not less than lambda_lb, and if so, go
                # half the way to lambda_lb.
                if lambda < lambda_lb
                    lambda = (lambda_previous + lambda_lb) / 2
                end

                if abs(lambda - lambda_previous) < tolerance
                    reached_solution = true
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
* `delta_min`, the smallest allowable trust region radius. Optimization halts if the updated radius is smaller than this value. Defaults to `sqrt(eps(Float64))`.
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
    delta_min::Real = sqrt(eps(Float64)),
    eta::Real = 0.1,
    rho_lower::Real = 0.25,
    rho_upper::Real = 0.75,
    use_fg::Bool = true,
)
    NewtonTrustRegion(promote(initial_delta, delta_hat, delta_min, eta, rho_lower, rho_upper)..., use_fg)
end

Base.summary(io::IO, ::NewtonTrustRegion) = print(io, "Newton's Method (Trust Region)")

mutable struct NewtonTrustRegionState{Tx,T,G} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    g_previous::G
    f_x_previous::T
    s::Tx
    x_cache::Tx
    g_cache::G
    hard_case::Bool
    reached_subproblem_solution::Bool
    interior::Bool
    delta::T
    lambda::T
    eta::T
    rho::T
end

function initial_state(method::NewtonTrustRegion, options, d, initial_x)
    T = eltype(initial_x)
    n = length(initial_x)
    # Keep track of trust region sizes
    delta = copy(method.initial_delta)

    # Record attributes of the subproblem in the trace.
    hard_case = false
    reached_subproblem_solution = true
    interior = true
    lambda = NaN

    NLSolversBase.value_gradient_hessian!!(d, initial_x)

    NewtonTrustRegionState(
        copy(initial_x), # Maintain current state in state.x
        copy(initial_x), # Maintain previous state in state.x_previous
        copy(gradient(d)), # Store previous gradient in state.g_previous
        T(NaN), # Store previous f in state.f_x_previous
        similar(initial_x), # Maintain current search direction in state.s
        fill!(similar(initial_x), NaN), # For resetting the state if a step is rejected
        fill!(method.use_fg ? similar(gradient(d)) : empty(gradient(d)), NaN), # For resetting the gradient if a step is rejected
        hard_case,
        reached_subproblem_solution,
        interior,
        T(delta),
        T(lambda),
        T(method.eta), # eta
        zero(T),
    ) # rho
end


function update_state!(d, state::NewtonTrustRegionState, method::NewtonTrustRegion)
    T = eltype(state.x)
    # Find the next step direction.
    m, state.interior, state.lambda, state.hard_case, state.reached_subproblem_solution =
        solve_tr_subproblem!(gradient(d), NLSolversBase.hessian(d), state.delta, state.s)

    # Maintain a record of previous position
    copyto!(state.x_cache, state.x)
    f_cache = value(d)

    # Update current position
    state.x .+= state.s
    # Update the function value and gradient
    if method.use_fg
        copyto!(state.g_cache, gradient(d))
        value_gradient!(d, state.x)
    else
        value!(d, state.x)
    end
    # Update the trust region size based on the discrepancy between
    # the predicted and actual function values.  (Algorithm 4.1 in N&W (2006))
    f_x_diff = f_cache - value(d)
    if abs(m) <= eps(T)
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
        state.delta = norm(state.s) / 4
    elseif state.rho < method.rho_lower
        state.delta /= 4
    elseif (state.rho > method.rho_upper) && !state.interior
        state.delta = min(2 * state.delta, method.delta_hat)
    end

    # Update/reset gradients and function values
    if accept_step
        if method.use_fg
            hessian!(d, state.x)
        else
            NLSolversBase.gradient_hessian!!(d, state.x)
        end
    else
        # Reset state
        copyto!(state.x, state.x_cache)

        # Reset objective function
        copyto!(d.x_f, state.x_cache)
        d.F = f_cache
        if method.use_fg
            copyto!(d.x_df, state.x_cache)
            copyto!(d.DF, state.g_cache)
        end
    end

    false
end

function assess_convergence(state::NewtonTrustRegionState, d, options::Options)
    x_converged, f_converged, g_converged, converged, f_increased =
        false, false, false, false, false
    if state.rho > state.eta
        # Accept the point and check convergence
        x_converged, f_converged, g_converged, f_increased = assess_convergence(
            state.x,
            state.x_previous,
            value(d),
            state.f_x_previous,
            gradient(d),
            options.x_abstol,
            options.f_reltol,
            options.g_abstol,
        )
    end
    x_converged, f_converged, g_converged, f_increased
end

function trace!(
    tr,
    d,
    state::NewtonTrustRegionState,
    iteration::Integer,
    method::NewtonTrustRegion,
    options::Options,
    curr_time = time(),
)
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["h(x)"] = copy(NLSolversBase.hessian(d))
        dt["delta"] = copy(state.delta)
        dt["interior"] = state.interior
        dt["hard case"] = state.hard_case
        dt["reached_subproblem_solution"] = state.reached_subproblem_solution
        dt["lambda"] = state.lambda
    end
    g_norm = norm(gradient(d), Inf)
    update!(
        tr,
        iteration,
        value(d),
        g_norm,
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback,
    )
end
