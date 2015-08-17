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
            if extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
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

function accelerated_gradient_descent{T}(d::DifferentiableFunction,
                                         initial_x::Vector{T};
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

    # Maintain current intermediate state in y and previous intermediate state in y_previous
    y, y_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr
    gr = Array(T, n)

    # The current search direction
    s = Array(T, n)

    # Buffers for use in line search
    x_ls, gr_ls = Array(T, n), Array(T, n)

    # Store f(x) in f_x
    f_x_previous, f_x = NaN, d.fg!(x, gr)
    f_calls, g_calls = f_calls + 1, g_calls + 1

    # Keep track of step-sizes
    alpha = alphainit(one(T), x, gr, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace || callback != nothing
    @agdtrace

    # Assess types of convergence
    x_converged, f_converged, gr_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Search direction is always the negative gradient
        for i in 1:n
            @inbounds s[i] = -gr[i]
        end

        # Refresh the line search cache
        dphi0 = _dot(gr, s)
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          linesearch!(d, x, s, x_ls, gr_ls, lsr, alpha, mayterminate)
        f_calls, g_calls = f_calls + f_update, g_calls + g_update

        # Make one move in the direction of the gradient
        copy!(y_previous, y)
        for i in 1:n
            @inbounds y[i] = x_previous[i] + alpha * s[i]
        end

        # Record previous state
        copy!(x_previous, x)

        # Update current position with Nesterov correction
        scaling = (iteration - 1) / (iteration + 2)
        for i in 1:n
            @inbounds x[i] = y[i] + scaling * (y[i] - y_previous[i])
        end

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

        @agdtrace
    end

    return MultivariateOptimizationResults("Accelerated Gradient Descent",
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
