# Notational note
# JMW's dx_history <=> NW's S
# JMW's dgr_history <=> NW's Y

# Here alpha is a cache that parallels betas
# It is not the step-size
# q is also a cache
function twoloop!(s::Vector,
                  gr::Vector,
                  rho::Vector,
                  dx_history::Matrix,
                  dgr_history::Matrix,
                  m::Integer,
                  iteration::Integer,
                  alpha::Vector,
                  q::Vector)
    # Count number of parameters
    n = length(s)

    # Determine lower and upper bounds for loops
    lower = iteration - m
    upper = iteration - 1

    # Copy gr into q for backward pass
    copy!(q, gr)

    # Backward pass
    for index in upper:-1:lower
        if index < 1
            continue
        end
        i = mod1(index, m)
        @inbounds alpha[i] = rho[i] * dot(dx_history[:, i], q)
        for j in 1:n
            @inbounds q[j] -= alpha[i] * dgr_history[j, i]
        end
    end

    # Copy q into s for forward pass
    copy!(s, q)

    # Forward pass
    for index in lower:1:upper
        if index < 1
            continue
        end
        i = mod1(index, m)
        @inbounds beta = rho[i] * dot(dgr_history[:, i], s)
        for j in 1:n
            @inbounds s[j] += dx_history[j, i] * (alpha[i] - beta)
        end
    end

    # Negate search direction
    for i in 1:n
        @inbounds s[i] = -1.0 * s[i]
    end

    return
end

macro lbfgstrace()
    esc(quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
                dt["Current step size"] = alpha
            end
            grnorm = norm(gr, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end)
end

function l_bfgs{T}(d::Union{DifferentiableFunction,
                            TwiceDifferentiableFunction},
                   initial_x::Vector{T};
                   constraints::AbstractConstraints = ConstraintsNone(),
                   interior::Bool = false,
                   m::Integer = 10,
                   xtol::Real = 1e-32,
                   ftol::Real = 1e-8,
                   grtol::Real = 1e-8,
                   iterations::Integer = 1_000,
                   store_trace::Bool = false,
                   show_trace::Bool = false,
                   extended_trace::Bool = false,
                   linesearch!::Function = hz_linesearch!)

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr and previous gradient in gr_previous
    gr, gr_previous = Vector{T}(n), Vector{T}(n)

    # Store a history of changes in position and gradient
    rho = Vector{T}(m)
    dx_history, dgr_history = Matrix{T}(n, m), Matrix{T}(n, m)

    # The current search direction
    s = Vector{T}(n)

    # Buffers for use in line search
    x_ls, gr_ls = Vector{T}(n), Vector{T}(n)

    # Store f(x) in f_x
    f_x_previous, f_x = NaN, d.fg!(x, gr)
    f_calls, g_calls = f_calls + 1, g_calls + 1
    copy!(gr_previous, gr)

    # Keep track of step-sizes
    alpha = alphainit(one(T), x, gr, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Buffers for new entries of dx_history and dgr_history
    dx, dgr = Vector{T}(n), Vector{T}(n)

    # Buffers for use by twoloop!
    twoloop_q, twoloop_alpha = Vector{T}(n), Vector{T}(m)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace
    @lbfgstrace

    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false

    # Iterate until convergence
    converged = false
    converged_iteration = typemin(Int)
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Determine the L-BFGS search direction
        twoloop!(s, gr, rho, dx_history, dgr_history, m, iteration,
                 twoloop_alpha, twoloop_q)

        # Refresh the line search cache
        dphi0 = vecdot(gr, s)
        if dphi0 >= 0
            # The search direction is not a descent direction. Something
            # is wrong, so restart the search-direction algorithm.
            # See also the "restart" below.
            iteration = 1
            for i = 1:n
                @inbounds s[i] = -gr[i]
            end
            dphi0 = vecdot(gr, s)
        end
        dphi0 == 0 && break   # we're at a stationary point
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        alphamax = interior ? toedge(x, s, constraints) : convert(T,Inf)

        # Pick the initial step size (HZ #I1-I2). Even though we might
        # guess alpha=1 for l_bfgs most of the time, testing suggests
        # this is still usually a good idea (fewer total function and
        # gradient evaluations).
        alpha, mayterminate, f_update, g_update =
            alphatry(alpha, d, x, s, x_ls, gr_ls, lsr, constraints, alphamax)
        f_calls, g_calls = f_calls + f_update, g_calls + g_update

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          linesearch!(d, x, s, x_ls, gr_ls, lsr, alpha, mayterminate, constraints, alphamax)
        f_calls, g_calls = f_calls + f_update, g_calls + g_update

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position
        for i in 1:n
            @inbounds dx[i] = alpha * s[i]
            @inbounds x[i] = x[i] + dx[i]
        end

        # Maintain a record of the previous gradient
        copy!(gr_previous, gr)

        # Update the function value and gradient
        f_x_previous, f_x = NaN, d.fg!(x, gr)
        f_calls, g_calls = f_calls + 1, g_calls + 1

        # Measure the change in the gradient
        for i in 1:n
            @inbounds dgr[i] = gr[i] - gr_previous[i]
        end

        # Update the L-BFGS history of positions and gradients
        rho_iteration = 1 / vecdot(dx, dgr)
        if isinf(rho_iteration)
            # TODO: Introduce a formal error? There was a warning here previously
            break
        end
        dx_history[:, mod1(iteration, m)] = dx
        dgr_history[:, mod1(iteration, m)] = dgr
        rho[mod1(iteration, m)] = rho_iteration

        x_converged,
        f_converged,
        gr_converged,
        converged = assess_convergence(x,
                                       x_previous,
                                       f_x,
                                       f_x_previous,
                                       gr,
                                       xtol,
                                       ftol,
                                       grtol)

        @lbfgstrace

        # If we think we've converged, restart and make sure we don't
        # have another long descent. This should catch the case where
        # dphi < 0 "by a hair," meaning that the chosen search direction
        # happened to be nearly orthogonoal to the gradient.
        if converged
            if iteration <= 3 || converged_iteration >= 0
                if iteration <= max(m, 3)
                    break  # we converged quickly after previous restart
                end
                converged_iteration += iteration
            else
                converged_iteration = iteration
            end
            converged = false
            iteration = 0  # the next search direction will be -gr
        end
    end

    if converged_iteration >= 0
        iteration += converged_iteration
    end
    return MultivariateOptimizationResults("L-BFGS",
                                           initial_x,
                                           x,
                                           @compat(Float64(f_x)),
                                           iteration,
                                           iteration == iterations,
                                           x_converged,
                                           xtol,
                                           f_converged,
                                           ftol,
                                           gr_converged,
                                           grtol,
                                           tr,
                                           f_calls,
                                           g_calls)
end
