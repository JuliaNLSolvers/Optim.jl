macro newton_tr_trace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
                dt["h(x)"] = copy(H)
                dt["delta"] = copy(delta)
                dt["interior"] = interior
            end
            grnorm = norm(gr, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    o.store_trace,
                    o.show_trace,
                    o.show_every,
                    o.callback)
        end
    end
end


# Check whether we are in the "hard case".
#
# Args:
#  H_eigv: The eigenvalues of H, low to high
#  qg: The inner product of the eigenvalues and the gradient in the same order
#
# Returns:
#  hard_case: Whether it is a candidate for the hard case
#  lambda_1_multiplicity: The number of times the lowest eigenvalue is repeated,
#                         which is only correct if hard_case is true.
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

    hard_case, lambda_index - 1
end

# Choose a point in the trust region for the next step using
# the interative (nearly exact) method of section 4.3 of Nocedal and Wright.
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
function solve_tr_subproblem!{T}(gr::Vector{T},
                                 H::Matrix{T},
                                 delta::T,
                                 s::Vector{T};
                                 tolerance=1e-10,
                                 max_iters=5)
    n = length(gr)
    delta_sq = delta^2

    @assert n == length(s)
    @assert (n, n) == size(H)
    @assert max_iters >= 1

    # Note that currently the eigenvalues are only sorted if H is perfectly
    # symmetric.  (Julia issue #17093)
    H_eig = eigfact(Symmetric(H))
    min_H_ev, max_H_ev = H_eig[:values][1], H_eig[:values][n]
    H_ridged = copy(H)

    # Cache the inner products between the eigenvectors and the gradient.
    qg = Array(T, n)
    for i=1:n
        qg[i] = vecdot(H_eig[:vectors][:, i], gr)
    end

    # Function 4.39 in N&W
    function p_sq_norm(lambda, min_i)
        p_sum = 0.
        for i = min_i:n
            p_sum += qg[i]^2 / (lambda + H_eig[:values][i])^2
        end
        p_sum
    end

    if min_H_ev >= 1e-8 && p_sq_norm(0.0, 1) <= delta_sq
        # No shrinkage is necessary: -(H \ gr) is the minimizer
        interior = true
        s[:] = -(H_eig[:vectors] ./ H_eig[:values]') * H_eig[:vectors]' * gr
        lambda = 0.0
    else
        interior = false

        # The hard case is when the gradient is orthogonal to all
        # eigenvectors associated with the lowest eigenvalue.
        hard_case_candidate, min_H_ev_multiplicity =
            check_hard_case_candidate(H_eig[:values], qg)

        # Solutions smaller than this lower bound on lambda are not allowed:
        # they don't ridge H enough to make H_ridge PSD.
        lambda_lb = -min_H_ev + max(1e-4, abs(min_H_ev) * 1e-4)
        lambda = lambda_lb

        hard_case = false
        if hard_case_candidate
            # The "hard case". lambda is taken to be -min_H_ev and we only need
            # to find a multiple of an orthogonal eigenvector that lands the
            # iterate on the boundary.

            # Formula 4.45 in N&W
            p_lambda2 = p_sq_norm(lambda, min_H_ev_multiplicity + 1)
            if p_lambda2 > delta_sq
                # Then we can simply solve using root finding.
                # Set a starting point between the minimum and largest eigenvalues.
                lambda = lambda_lb + 0.01 * (max_H_ev - lambda_lb)
            else
                hard_case = true
                tau = sqrt(delta_sq - p_lambda2)

                # I don't think it matters which eigenvector we pick so take
                # the first.
                for i=1:n
                    s[i] = tau * H_eig[:vectors][i, 1]
                    for k=(min_H_ev_multiplicity + 1):n
                        s[i] = s[i] +
                               qg[k] * H_eig[:vectors][i, k] / (H_eig[:values][k] + lambda)
                    end
                end
            end
        end

        if !hard_case
            # Algorithim 4.3 of N&W, with s insted of p_l for consistency with
            # Optim.jl

            for i=1:n
                H_ridged[i, i] = H[i, i] + lambda
            end

            for iter in 1:max_iters
                lambda_previous = lambda

                # Version 0.5 requires an exactly symmetric matrix, but
                # version 0.4 does not have this function signature for chol().
                R = VERSION < v"0.5-" ? chol(H_ridged): chol(Hermitian(H_ridged))
                s[:] = -R \ (R' \ gr)
                q_l = R' \ s
                norm2_s = vecdot(s, s)
                lambda_update = norm2_s * (sqrt(norm2_s) - delta) / (delta * vecdot(q_l, q_l))
                lambda += lambda_update

                # Check that lambda is not less than lambda_lb, and if so, go
                # half the way to lambda_lb.
                if lambda < (lambda_lb + 1e-8)
                    lambda = 0.5 * (lambda_previous - lambda_lb) + lambda_lb
                end

                for i=1:n
                    H_ridged[i, i] = H[i, i] + lambda
                end

                if abs(lambda - lambda_previous) < tolerance
                    break
                end
            end
        end
    end

    m = vecdot(gr, s) + 0.5 * vecdot(s, H_ridged * s)

    return m, interior, lambda
end


immutable NewtonTrustRegion{T <: Real} <: Optimizer
    initial_delta::T
    delta_hat::T
    eta::T
    rho_lower::T
    rho_upper::T
end

NewtonTrustRegion(; initial_delta::Real=1.0,
                    delta_hat::Real = 100.0,
                    eta::Real = 0.1,
                    rho_lower::Real = 0.25,
                    rho_upper::Real = 0.75) =
  NewtonTrustRegion(initial_delta, delta_hat, eta, rho_lower, rho_upper)


function optimize{T}(d::TwiceDifferentiableFunction,
                     initial_x::Vector{T},
                     mo::NewtonTrustRegion,
                     o::OptimizationOptions)

    @assert(mo.delta_hat > 0, "delta_hat must be strictly positive")
    @assert(0 < mo.initial_delta < mo.delta_hat, "delta must be in (0, delta_hat)")
    @assert(0 <= mo.eta < 0.25, "eta must be in [0, 0.25)")
    @assert(mo.rho_lower < mo.rho_upper, "must have rho_lower < rho_upper")
    @assert(mo.rho_lower >= 0.)

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 1

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr
    gr = Array(T, n)

    # The current search direction
    s = Array(T, n)

    # Store f(x), the function value, in f_x
    f_x_previous, f_x = NaN, d.fg!(x, gr)

    # We need to store the previous gradient in case we reject a step.
    gr_previous = copy(gr)

    f_calls, g_calls = f_calls + 1, g_calls + 1

    # Store the hessian in H
    H = Array(T, n, n)
    d.h!(x, H)

    # Keep track of trust region sizes
    delta = copy(mo.initial_delta)

    # Record whether the point is interior in the trace.
    interior = false

    # Trace the history of states visited
    tr = OptimizationTrace(mo)
    tracing = o.store_trace || o.show_trace || o.extended_trace || o.callback != nothing
    @newton_tr_trace

    # Assess multiple types of convergence
    x_converged, f_converged, g_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration <= o.iterations
        # Find the next step direction.
        m, interior = solve_tr_subproblem!(gr, H, delta, s)

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position
        for i in 1:n
            @inbounds x[i] = x[i] + s[i]
        end

        # Update the function value and gradient
        copy!(gr_previous, gr)
        f_x_previous, f_x = f_x, d.fg!(x, gr)
        f_calls, g_calls = f_calls + 1, g_calls + 1

        # Update the trust region size based on the discrepancy between
        # the predicted and actual function values.  (Algorithm 4.1 in N&W)
        f_x_diff = f_x_previous - f_x
        if m == 0
            # This should only happen if the step is zero, in which case
            # we should accept the step and assess_convergence().
            @assert(f_x_diff == 0,
                    "m == 0 but the actual function change ($f_x_diff) is nonzero")
            rho = 1.0
        elseif m > 0
            # This can happen if the trust region radius is too large and the
            # Hessian is not positive definite.  We should shrink the trust
            # region.
            rho = mo.rho_lower - 1.0
        else
            rho = f_x_diff / (0 - m)
        end

        if rho < mo.rho_lower
            delta *= 0.25
        elseif (rho > mo.rho_upper) && (!interior)
            delta = min(2 * delta, mo.delta_hat)
        else
            # else leave delta unchanged.
        end

        if rho > mo.eta
            # Accept the point and check convergence

            x_converged,
            f_converged,
            g_converged,
            converged = assess_convergence(x,
                                           x_previous,
                                           f_x,
                                           f_x_previous,
                                           gr,
                                           o.x_tol,
                                           o.f_tol,
                                           o.g_tol)
            if !converged
                # Only compute the next Hessian if we haven't converged
                d.h!(x, H)
            end
        else
            # The improvement is too small and we won't take it.

            # If you reject an interior solution, make sure that the next
            # delta is smaller than the current step.  Otherwise you waste
            # steps reducing delta by constant factors while each solution
            # will be the same.
            x_diff = x - x_previous
            delta = 0.25 * sqrt(vecdot(x_diff, x_diff))

            f_x = f_x_previous
            copy!(x, x_previous)
            copy!(gr, gr_previous)

        end

        # Increment the number of steps we've had to perform
        iteration += 1

        @newton_tr_trace
    end

    return MultivariateOptimizationResults("Newton's Method with Trust Region",
                                           initial_x,
                                           x,
                                           Float64(f_x),
                                           iteration,
                                           iteration == iterations,
                                           x_converged,
                                           o.x_tol,
                                           f_converged,
                                           o.f_tol,
                                           g_converged,
                                           o.g_tol,
                                           tr,
                                           f_calls,
                                           g_calls)
end
