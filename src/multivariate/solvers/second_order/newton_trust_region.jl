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
function calc_p!(lambda::T, min_i, n, qg, H_eig, p) where T
    fill!( p, zero(T) )
    for i = min_i:n
        p[:] -= qg[i] / (H_eig.values[i] + lambda) * H_eig.vectors[:, i]
    end
    return nothing
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
function solve_tr_subproblem!(gr,
                              H,
                              delta,
                              s;
                              tolerance=1e-10,
                              max_iters=5)
    T = eltype(gr)
    n = length(gr)
    delta_sq = delta^2

    @assert n == length(s)
    @assert (n, n) == size(H)
    @assert max_iters >= 1

    # Note that currently the eigenvalues are only sorted if H is perfectly
    # symmetric.  (Julia issue #17093)
    H_eig = eigen(Symmetric(H))
    min_H_ev, max_H_ev = H_eig.values[1], H_eig.values[n]
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
        hard_case_candidate, min_i =
            check_hard_case_candidate(H_eig.values, qg)

        # Solutions smaller than this lower bound on lambda are not allowed:
        # they don't ridge H enough to make H_ridge PSD.
        lambda_lb = -min_H_ev + max(1e-8, 1e-8 * (max_H_ev - min_H_ev))
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
                s[:] = -s + tau * H_eig.vectors[:, 1]
            end
        end

        if !hard_case
            # Algorithim 4.3 of N&W (2006), with s insted of p_l for consistency
            # with Optim.jl

            reached_solution = false
            for iter in 1:max_iters
                lambda_previous = lambda

                for i=1:n
                    H_ridged[i, i] = H[i, i] + lambda
                end

                R = cholesky(Hermitian(H_ridged)).U
                s[:] = -R \ (R' \ gr)
                q_l = R' \ s
                norm2_s = dot(s, s)
                lambda_update = norm2_s * (sqrt(norm2_s) - delta) / (delta * dot(q_l, q_l))
                lambda += lambda_update

                # Check that lambda is not less than lambda_lb, and if so, go
                # half the way to lambda_lb.
                if lambda < lambda_lb
                    lambda = 0.5 * (lambda_previous - lambda_lb) + lambda_lb
                end

                if abs(lambda - lambda_previous) < tolerance
                    reached_solution = true
                    break
                end
            end
        end
    end

    m = dot(gr, s) + 0.5 * dot(s, H * s)

    return m, interior, lambda, hard_case, reached_solution
end

struct NewtonTrustRegion{T <: Real} <: SecondOrderOptimizer
    initial_delta::T
    delta_hat::T
    eta::T
    rho_lower::T
    rho_upper::T
end

"""
# NewtonTrustRegion
## Constructor
```julia
NewtonTrustRegion(; initial_delta = 1.0,
                    delta_hat = 100.0,
                    eta = 0.1,
                    rho_lower = 0.25,
                    rho_upper = 0.75)
```

The constructor has 5 keywords:
* `initial_delta`, the starting trust region radius
* `delta_hat`, the largest allowable trust region radius
* `eta`, when `rho` is at least `eta`, accept the step
* `rho_lower`, when `rho` is less than `rho_lower`, shrink the trust region
* `rho_upper`, when `rho` is greater than `rho_upper`, grow the trust region

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
NewtonTrustRegion(; initial_delta::Real = 1.0,
                    delta_hat::Real = 100.0,
                    eta::Real = 0.1,
                    rho_lower::Real = 0.25,
                    rho_upper::Real = 0.75) =
                    NewtonTrustRegion(initial_delta, delta_hat, eta, rho_lower, rho_upper)

Base.summary(::NewtonTrustRegion) = "Newton's Method (Trust Region)"

mutable struct NewtonTrustRegionState{Tx, T, G} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    g_previous::G
    f_x_previous::T
    s::Tx
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
    # Maintain current gradient in gr
    @assert(method.delta_hat > 0, "delta_hat must be strictly positive")
    @assert(0 < method.initial_delta < method.delta_hat, "delta must be in (0, delta_hat)")
    @assert(0 <= method.eta < method.rho_lower, "eta must be in [0, rho_lower)")
    @assert(method.rho_lower < method.rho_upper, "must have rho_lower < rho_upper")
    @assert(method.rho_lower >= 0.)
    # Keep track of trust region sizes
    delta = copy(method.initial_delta)

    # Record attributes of the subproblem in the trace.
    hard_case = false
    reached_subproblem_solution = true
    interior = true
    lambda = NaN

    value_gradient!!(d, initial_x)
    hessian!!(d, initial_x)

    NewtonTrustRegionState(copy(initial_x), # Maintain current state in state.x
                         copy(initial_x), # Maintain previous state in state.x_previous
                         copy(gradient(d)), # Store previous gradient in state.g_previous
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         hard_case,
                         reached_subproblem_solution,
                         interior,
                         T(delta),
                         T(lambda),
                         T(method.eta), # eta
                         zero(T)) # rho
end


function update_state!(d, state::NewtonTrustRegionState, method::NewtonTrustRegion)
    T = eltype(state.x)
    # Find the next step direction.
    m, state.interior, state.lambda, state.hard_case, state.reached_subproblem_solution =
        solve_tr_subproblem!(gradient(d), NLSolversBase.hessian(d), state.delta, state.s)

    # Maintain a record of previous position
    copyto!(state.x_previous, state.x)
    state.f_x_previous  = value(d)
    copyto!(state.g_previous, gradient(d))

    # Update current position
    state.x .+= state.s

    # Update the function value and gradient
    value_gradient!(d, state.x)

    # Update the trust region size based on the discrepancy between
    # the predicted and actual function values.  (Algorithm 4.1 in N&W (2006))
    f_x_diff = state.f_x_previous - value(d)
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
        state.rho = f_x_diff / (0 - m)
    end

    if state.rho < method.rho_lower
        state.delta *= 0.25
    elseif (state.rho > method.rho_upper) && (!state.interior)
        state.delta = min(2 * state.delta, method.delta_hat)
    else
        # else leave delta unchanged.
    end

    if state.rho <= state.eta
        # The improvement is too small and we won't take it.

        # If you reject an interior solution, make sure that the next
        # delta is smaller than the current step. Otherwise you waste
        # steps reducing delta by constant factors while each solution
        # will be the same.
        x_diff = state.x - state.x_previous
        delta = 0.25 * norm(x_diff)

        d.F = state.f_x_previous
        copyto!(state.x, state.x_previous)
        copyto!(gradient(d), state.g_previous)
    end

    false
end

function assess_convergence(state::NewtonTrustRegionState, d, options::Options)
    x_converged, f_converged, g_converged, converged, f_increased = false, false, false, false, false
    if state.rho > state.eta
        # Accept the point and check convergence
        x_converged,
        f_converged,
        g_converged,
        converged,
        f_increased = assess_convergence(state.x,
                                       state.x_previous,
                                       value(d),
                                       state.f_x_previous,
                                       gradient(d),
                                       options.x_abstol,
                                       options.f_reltol,
                                       options.g_abstol)
    end
    x_converged, f_converged, g_converged, converged, f_increased
end

function trace!(tr, d, state, iteration, method::NewtonTrustRegion, options, curr_time=time())
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
