function gradient_descent_trace!(tr::OptimizationTrace,
                                 x::Vector,
                                 f_x::Real,
                                 gradient::Vector,
                                 iteration::Integer,
                                 store_trace::Bool,
                                 show_trace::Bool)
    dt = Dict()
    dt["g(x)"] = copy(gradient)
    dt["|g(x)|"] = norm(gradient, Inf)
    os = OptimizationState(copy(x), f_x, iteration, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function gradient_descent(d::DifferentiableFunction,
                          initial_x::Vector;
                          tolerance::Real = 1e-8,
                          iterations::Integer = 1_000,
                          store_trace::Bool = false,
                          show_trace::Bool = false,
                          line_search!::Function = backtracking_line_search!)

    # Keep a copy of the initial state of the system
    x = copy(initial_x)

    # Count the number of gradient descent steps we perform
    iteration = 0

    # Track calls to f and g!
    f_calls, g_calls = 0, 0

    # Allocate vectors for re-use by gradient calls
    n = length(x)
    gradient = Array(Float64, n)

    # Allocate vector for step direction
    p = Array(Float64, n)

    # Allocate vectors for line search
    ls_x = Array(Float64, n)
    ls_gradient = Array(Float64, n)

    # Maintain current value of f and g
    f_calls += 1
    g_calls += 1
    f_x = d.fg!(x, gradient)

    # Show trace
    tr = OptimizationTrace()
    if store_trace || show_trace
        gradient_descent_trace!(tr,
                                x,
                                f_x,
                                gradient,
                                iteration,
                                store_trace,
                                show_trace)
    end

    # Monitor convergence
    converged = false

    # Iterate until the norm of the gradient is within tolerance of zero
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration = iteration + 1

        # Use a back-tracking line search to select a step-size
        #  Direction is always negative gradient
        for i in 1:n
            p[i] = -gradient[i]
        end
        step_size, f_update, g_update =
          line_search!(d, x, p, ls_x, ls_gradient)
        f_calls += f_update
        g_calls += g_update

        # Move in the direction of the gradient
        for i in 1:n
            x[i] = x[i] - step_size * gradient[i]
        end

        # Update the function value and gradient
        f_calls += 1
        g_calls += 1
        f_x = d.fg!(x, gradient)

        # Assess convergence
        if norm(gradient, Inf) <= tolerance
            converged = true
        end

        # Show trace
        if store_trace || show_trace
            gradient_descent_trace!(tr,
                                    x,
                                    f_x,
                                    gradient,
                                    iteration,
                                    store_trace,
                                    show_trace)
        end
    end

    OptimizationResults("Gradient Descent",
                        initial_x,
                        x,
                        f_x,
                        iteration,
                        converged,
                        tr,
                        f_calls,
                        g_calls)
end
