function newton_trace!(tr::OptimizationTrace,
                       x::Vector,
                       f_x::Real,
                       i::Integer,
                       gradient::Vector,
                       H::Matrix,
                       store_trace::Bool,
                       show_trace::Bool)
    dt = Dict()
    dt["g(x)"] = copy(gradient)
    dt["h(x)"] = copy(H)
    os = OptimizationState(x, f_x, i, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function newton(d::TwiceDifferentiableFunction,
                initial_x::Vector;
                tolerance::Real = 1e-8,
                iterations::Integer = 1_000,
                store_trace::Bool = false,
                show_trace::Bool = false)

    # Maintain a record of the initial state
    x = copy(initial_x)

    # Count function and gradient calls
    f_calls = 0
    g_calls = 0

    # Count number of parameters
    n = length(x)

    # Cache the values of f, g and h
    gradient = Array(Float64, n)
    f_x = d.fg!(x, gradient)
    f_calls += 1
    g_calls += 1
    H = Array(Float64, n, n)
    d.h!(x, H)

    # Allocate buffers
    ls_x = Array(Float64, n)
    ls_gradient = Array(Float64, n)

    # Count iterations
    iteration = 0

    # Maintain a trace of the system
    tr = OptimizationTrace()
    if store_trace || show_trace
        newton_trace!(tr, x, f_x, iteration, gradient, H, store_trace, show_trace)
    end

    # Track convergence
    converged = false

    # Determine direction of line search
    dx = -H \ gradient

    while !converged && iteration < iterations
        # Update the iteration counter
        iteration += 1

        # Select a step size
        step_size, f_up, g_up =
          backtracking_line_search!(d, x, dx, ls_x, ls_gradient)
        f_calls += f_up
        g_calls += g_up

        # Update our position
        for i in 1:n
            x[i] += step_size * dx[i]
        end

        # Cache values again
        f_x = d.fg!(x, gradient)
        d.h!(x, H)
        f_calls += 1
        g_calls += 1

        # Select a search direction
        dx = -H \ gradient

        # Assess convergence
        if norm(gradient, Inf) <= tolerance
           converged = true
        end

        # Show state of the system
        if store_trace || show_trace
            newton_trace!(tr, x, f_x, iteration, gradient, H, store_trace, show_trace)
        end
    end

    OptimizationResults("Newton's Method",
                        initial_x,
                        x,
                        f_x,
                        iteration,
                        converged,
                        tr,
                        f_calls,
                        g_calls)
end
