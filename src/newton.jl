function newton_trace!(tr::OptimizationTrace,
                       x::Vector,
                       f_x::Real,
                       gr::Vector,
                       H::Matrix,
                       iteration::Integer,
                       store_trace::Bool,
                       show_trace::Bool)
    dt = Dict()
    dt["g(x)"] = copy(gr)
    dt["h(x)"] = copy(H)
    os = OptimizationState(copy(x), f_x, iteration, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function newton{T}(d::TwiceDifferentiableFunction,
                   initial_x::Vector{T};
                   xtol::Real = 1e-32,
                   ftol::Real = 1e-32,
                   grtol::Real = 1e-8,
                   iterations::Integer = 1_000,
                   store_trace::Bool = false,
                   show_trace::Bool = false,
                   linesearch!::Function = hz_linesearch!)

    # Maintain current state in x and previous state in x_previous
    x = copy(initial_x)
    x_previous = copy(initial_x)

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
    # TODO: Try to avoid re-allocating s
    s = Array(Float64, n)

    # Buffers for use in line search
    x_ls = Array(Float64, n)
    gr_ls = Array(Float64, n)

    # Store f(x) in f_x
    f_x = d.fg!(x, gr)
    f_calls += 1
    g_calls += 1

    # Store h(x) in H
    H = Array(Float64, n, n)
    d.h!(x, H)

    # Store the history of function values
    f_values = Array(T, iterations + 1)
    fill!(f_values, nan(T))
    f_values[iteration + 1] = f_x

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
        newton_trace!(tr, x, f_x, gr, H,
                      iteration, store_trace, show_trace)
    end

    # Iterate until convergence
    x_converged = false
    f_converged = false
    gr_converged = false
    converged = false
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Search direction is always the negative gradient divided by H
        # TODO: Do this calculation in place
        s[:] = -(H \ gr)

        # Refresh the line search cache
        dphi0 = dot(gr, s)
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        # TODO: Fix f_update, g_update
        alpha, f_update, g_update =
          linesearch!(d, x, s, x_ls, gr_ls, lsr, alpha, mayterminate)
        f_calls += f_update
        g_calls += g_update

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position
        for i in 1:n
            x[i] = x[i] + alpha * s[i]
        end

        # Update the function value and gradient
        f_x = d.fg!(x, gr)
        f_calls += 1
        g_calls += 1
        f_values[iteration + 1] = f_x

        # Update the Hessian
        d.h!(x, H)

        # Assess convergence
        deltax = 0.0
        for i in 1:n
            diff = abs(x[i] - x_previous[i])
            if diff > deltax
                deltax = diff
            end
        end
        if deltax < xtol
            x_converged = true
        end
        if abs(f_values[iteration + 1] - f_values[iteration]) < ftol
            f_converged = true
        end
        if norm(gr, Inf) < grtol
            gr_converged = true
        end
        converged = x_converged || f_converged || gr_converged

        # Show trace
        if tracing
            newton_trace!(tr, x, f_x, gr, H,
                          iteration, store_trace, show_trace)
        end
    end

    OptimizationResults("Newton's Method",
                        initial_x,
                        x,
                        f_x,
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
                        g_calls,
                        f_values[1:(iteration + 1)])
end
