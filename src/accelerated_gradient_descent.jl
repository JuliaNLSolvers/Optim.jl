# http://stronglyconvex.com/blog/accelerated-gradient-descent.html
# TODO: Need to specify alphamax on each iteration
# Flip notation relative to Duckworth
# Start with x_{0}
# y_{t} = x_{t - 1} - alpha g(x_{t - 1})
# If converged, return y_{t}
# x_{t} = y_{t} + (t - 1.0) / (t + 2.0) * (y_{t} - y_{t - 1})

function accelerated_gradient_descent_trace!(tr::OptimizationTrace,
                                             x::Vector,
                                             f_x::Real,
                                             gr::Vector,
                                             iteration::Integer,
                                             store_trace::Bool,
                                             show_trace::Bool)
    dt = Dict()
    dt["g(x)"] = copy(gr)
    dt["Maximum component of g(x)"] = norm(gr, Inf)
    os = OptimizationState(copy(x), f_x, iteration, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function accelerated_gradient_descent{T}(d::DifferentiableFunction,
                                         initial_x::Vector{T};
                                         tolerance::Real = 1e-8,
                                         iterations::Integer = 1_000,
                                         store_trace::Bool = false,
                                         show_trace::Bool = false,
                                         linesearch!::Function = hz_linesearch!)

    # Maintain current state in x and previous state in x_previous
    x_previous = copy(initial_x)
    x = copy(initial_x)

    # Maintain current intermediate state in y and previous intermediate state in y_previous
    y_previous = copy(initial_x)
    y = copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls = 0
    g_calls = 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr
    gr = Array(Float64, n)

    # The current search direction
    s = Array(Float64, n)

    # Buffers for use in line search
    x_ls = Array(Float64, n)
    gr_ls = Array(Float64, n)

    # Store f(x) in f_x
    f_x = d.fg!(x, gr)
    f_calls += 1
    g_calls += 1

    # Keep track of step-sizes
    alpha = alphainit(1.0, x, gr, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace
    if tracing
        accelerated_gradient_descent_trace!(tr, x, f_x, gr,
                                            iteration, store_trace, show_trace)
    end

    # Iterate until convergence
    converged = false
    while !converged && iteration <= iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Search direction is always the negative gradient
        for i in 1:n
            s[i] = -gr[i]
        end

        # Refresh the line search cache
        dphi0 = dot(gr, s)
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          linesearch!(d, x, s, x_ls, gr_ls, lsr, alpha, mayterminate)
        f_calls += f_update
        g_calls += g_update

        # Make one move in the direction of the gradient
        copy!(y_previous, y)
        for i in 1:n
            y[i] = x_previous[i] + alpha * s[i]
        end

        # Record previous state
        copy!(x_previous, x)

        # Update current position with Nesterov correction
        scaling = (iteration - 1.0) / (iteration + 2.0)
        for i in 1:n
            x[i] = y[i] + scaling * (y[i] - y_previous[i])
        end

        # Update the function value and gradient
        f_x = d.fg!(x, gr)
        f_calls += 1
        g_calls += 1

        # Assess convergence
        if norm(gr, Inf) <= tolerance
            converged = true
        end

        # Show trace
        if tracing
            accelerated_gradient_descent_trace!(tr, x, f_x, gr,
                                                iteration, store_trace, show_trace)
        end
    end

    OptimizationResults("Accelerated Gradient Descent",
                        initial_x,
                        x,
                        f_x,
                        iteration,
                        converged,
                        tr,
                        f_calls,
                        g_calls)
end
