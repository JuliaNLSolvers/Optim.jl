# Translation from our variables to Nocedal and Wright's
# JMW's dx <=> NW's s
# JMW's dg <=> NW' y

macro bfgstrace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(g)
                dt["~inv(H)"] = copy(invH)
            end
            g_norm = vecnorm(g, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    g_norm,
                    dt,
                    o.store_trace,
                    o.show_trace,
                    o.show_every,
                    o.callback)
        end
    end
end

immutable BFGS <: Optimizer
    linesearch!::Function
end

BFGS(; linesearch!::Function = hz_linesearch!) =
  BFGS(linesearch!)

function optimize{T}(d::DifferentiableFunction,
                     initial_x::Vector{T},
                     mo::BFGS,
                     o::OptimizationOptions;
                     initial_invH::Matrix = eye(length(initial_x)))
    # Print header if show_trace is set
    print_header(o)

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in g and previous gradient in g_previous
    g, g_previous = Array(T, n), Array(T, n)

    # Store the approximate inverse Hessian in invH
    invH = copy(initial_invH)

    # The current search direction
    s = Array(T, n)

    # Intermediate storage
    u = Array(T, n)

    # Buffers for use in line search
    x_ls, g_ls = Array(T, n), Array(T, n)

    # Store f(x) in f_x
    f_x_previous, f_x = NaN, d.fg!(x, g)
    f_calls, g_calls = f_calls + 1, g_calls + 1
    copy!(g_previous, g)

    # Keep track of step-sizes
    alpha = alphainit(one(T), x, g, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Maintain record of changes in position and gradient
    dx, dg = Array(T, n), Array(T, n)

    # Maintain a cached copy of the identity matrix
    I = eye(size(invH)...)

    # Trace the history of states visited
    tr = OptimizationTrace(mo)
    tracing = o.store_trace || o.show_trace || o.extended_trace || o.callback != nothing
    @bfgstrace

    # Assess multiple types of convergence
    x_converged, f_converged, g_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration < o.iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Set the search direction
        # Search direction is the negative gradient divided by the approximate Hessian
        A_mul_B!(s, invH, g)
        scale!(s, -1)

        # Refresh the line search cache
        dphi0 = vecdot(g, s)
        # If invH is not positive definite, reset it to I
        if dphi0 > 0.0
            copy!(invH, I)
            @simd for i in 1:n
                @inbounds s[i] = -g[i]
            end
            dphi0 = vecdot(g, s)
        end
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          mo.linesearch!(d, x, s, x_ls, g_ls, lsr, alpha, mayterminate)
        f_calls, g_calls = f_calls + f_update, g_calls + g_update

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position
        @simd for i in 1:n
            @inbounds dx[i] = alpha * s[i]
            @inbounds x[i] = x[i] + dx[i]
        end

        # Maintain a record of the previous gradient
        copy!(g_previous, g)

        # Update the function value and gradient
        f_x_previous, f_x = f_x, d.fg!(x, g)
        f_calls, g_calls = f_calls + 1, g_calls + 1

        x_converged,
        f_converged,
        g_converged,
        converged = assess_convergence(x,
                                       x_previous,
                                       f_x,
                                       f_x_previous,
                                       g,
                                       o.x_tol,
                                       o.f_tol,
                                       o.g_tol)

        # Measure the change in the gradient
        @simd for i in 1:n
            @inbounds dg[i] = g[i] - g_previous[i]
        end

        # Update the inverse Hessian approximation using Sherman-Morrison
        dx_dg = vecdot(dx, dg)
        if dx_dg == 0.0
            break
        end
        A_mul_B!(u, invH, dg)

        c1 = (dx_dg + vecdot(dg, u)) / (dx_dg * dx_dg)
        c2 = 1 / dx_dg

        # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
        for i in 1:n
            @simd for j in 1:n
                @inbounds invH[i, j] += c1 * dx[i] * dx[j] - c2 * (u[i] * dx[j] + u[j] * dx[i])
            end
        end

        @bfgstrace
    end

    return MultivariateOptimizationResults("BFGS",
                                           initial_x,
                                           x,
                                           Float64(f_x),
                                           iteration,
                                           iteration == o.iterations,
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
