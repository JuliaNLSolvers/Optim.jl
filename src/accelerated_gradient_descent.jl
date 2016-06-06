# http://stronglyconvex.com/blog/accelerated-gradient-descent.html
# TODO: Need to specify alphamax on each iteration
# Flip notation relative to Duckworth
# Start with x_{0}
# y_{t} = x_{t - 1} - alpha g(x_{t - 1})
# If converged, return y_{t}
# x_{t} = y_{t} + (t - 1.0) / (t + 2.0) * (y_{t} - y_{t - 1})

macro agdtrace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(g)
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

immutable AcceleratedGradientDescent <: Optimizer
    linesearch!::Function
end

AcceleratedGradientDescent(; linesearch!::Function = hz_linesearch!) =
  AcceleratedGradientDescent(linesearch!)

function optimize{T}(d::DifferentiableFunction,
                     initial_x::Vector{T},
                     mo::AcceleratedGradientDescent,
                     o::OptimizationOptions)
    # Print header if show_trace is set
    print_header(o)

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Maintain current intermediate state in y and previous intermediate state in y_previous
    y, y_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in g
    g = Array(T, n)

    # The current search direction
    s = Array(T, n)

    # Buffers for use in line search
    x_ls, g_ls = Array(T, n), Array(T, n)

    # Store f(x) in f_x
    f_x_previous, f_x = NaN, d.fg!(x, g)
    f_calls, g_calls = f_calls + 1, g_calls + 1

    # Keep track of step-sizes
    alpha = alphainit(one(T), x, g, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Trace the history of states visited
    tr = OptimizationTrace(mo)
    tracing = o.store_trace || o.show_trace || o.extended_trace || o.callback != nothing
    @agdtrace

    # Assess types of convergence
    x_converged, f_converged, g_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration < o.iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Search direction is always the negative gradient
        @simd for i in 1:n
            @inbounds s[i] = -g[i]
        end

        # Refresh the line search cache
        dphi0 = vecdot(g, s)
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          mo.linesearch!(d, x, s, x_ls, g_ls, lsr, alpha, mayterminate)
        f_calls, g_calls = f_calls + f_update, g_calls + g_update

        # Make one move in the direction of the gradient
        copy!(y_previous, y)
        @simd for i in 1:n
            @inbounds y[i] = x_previous[i] + alpha * s[i]
        end

        # Record previous state
        copy!(x_previous, x)

        # Update current position with Nesterov correction
        scaling = (iteration - 1) / (iteration + 2)
        @simd for i in 1:n
            @inbounds x[i] = y[i] + scaling * (y[i] - y_previous[i])
        end

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

        @agdtrace
    end

    return MultivariateOptimizationResults("Accelerated Gradient Descent",
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
