# Translation from our variables to Nocedal and Wright's
# JMW's dx <=> NW's s
# JMW's dgr <=> NW' y

macro bfgstrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
                dt["~inv(H)"] = copy(invH)
            end
            grnorm = norm(gr, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    store_trace,
                    show_trace,
                    show_every,
                    callback)
        end
    end
end

function bfgs{T}(d::Union(DifferentiableFunction,
                          TwiceDifferentiableFunction),
                 initial_x::Vector{T};
                 initial_invH::Matrix = eye(length(initial_x)),
                 xtol::Real = 1e-32,
                 ftol::Real = 1e-8,
                 grtol::Real = 1e-8,
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 callback = nothing,
                 show_every = 1,
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
    gr, gr_previous = Array(T, n), Array(T, n)

    # Store the approximate inverse Hessian in invH
    invH = copy(initial_invH)

    # The current search direction
    s = Array(T, n)

    # Intermediate storage
    u = Array(T, n)

    # Buffers for use in line search
    x_ls, gr_ls = Array(T, n), Array(T, n)

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

    # Maintain record of changes in position and gradient
    dx, dgr = Array(T, n), Array(T, n)

    # Maintain a cached copy of the identity matrix
    I = eye(size(invH)...)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace || callback != nothing
    @bfgstrace

    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Set the search direction
        # Search direction is the negative gradient divided by the approximate Hessian
        A_mul_B!(s, invH, gr)
        for i in 1:n
            @inbounds s[i] = -s[i]
        end

        # Refresh the line search cache
        dphi0 = _dot(gr, s)
        # If invH is not positive definite, reset it to I
        if dphi0 > 0.0
            copy!(invH, I)
            for i in 1:n
                @inbounds s[i] = -gr[i]
            end
            dphi0 = _dot(gr, s)
        end
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          linesearch!(d, x, s, x_ls, gr_ls, lsr, alpha, mayterminate)
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
        f_x_previous, f_x = f_x, d.fg!(x, gr)
        f_calls, g_calls = f_calls + 1, g_calls + 1

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

        # Measure the change in the gradient
        for i in 1:n
            @inbounds dgr[i] = gr[i] - gr_previous[i]
        end

        # Update the inverse Hessian approximation using Sherman-Morrison
        dx_dgr = _dot(dx, dgr)
        if dx_dgr == 0.0
            break
        end
        A_mul_B!(u, invH, dgr)

        c1 = (dx_dgr + _dot(dgr, u)) / (dx_dgr * dx_dgr)
        c2 = 1 / dx_dgr

        # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
        for i in 1:n
            for j in 1:n
                @inbounds invH[i, j] += c1 * dx[i] * dx[j] - c2 * (u[i] * dx[j] + u[j] * dx[i])
            end
        end

        @bfgstrace
    end

    return MultivariateOptimizationResults("BFGS",
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
