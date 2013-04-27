# http://stronglyconvex.com/blog/accelerated-gradient-descent.html

function accelerated_gradient_descent_trace!(tr::OptimizationTrace,
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

# Flip notation
# Start with x_{0}
# y_{t} = x_{t - 1} - alpha g(x_{t - 1})
# If converged, return y_{t}
# x_{t} = y_{t} + (t - 1.0) / (t + 2.0) * (y_{t} - y_{t - 1})

function accelerated_gradient_descent(d::DifferentiableFunction,
                                      initial_x::Vector;
                                      tolerance::Real = 1e-8,
                                      iterations::Integer = 1_000,
                                      store_trace::Bool = false,
                                      show_trace::Bool = false,
                                      line_search!::Function = backtracking_line_search!)

    # Keep a copy of the initial state of the system
    x_t1 = copy(initial_x)
    x_t = copy(initial_x)
    y_t1 = copy(initial_x)
    y_t = copy(initial_x)

    # Count the number of gradient descent steps we perform
    iteration = 0

    # Track calls to f and g!
    f_calls, g_calls = 0, 0

    # Allocate vectors for re-use by gradient calls
    n = length(initial_x)
    gradient = Array(Float64, n)

    # Maintain current value of f and g
    f_x = d.fg!(x_t, gradient)
    f_calls += 1
    g_calls += 1

    # Allocate vector for step direction
    p = Array(Float64, n)

    # Allocate vectors for line search
    ls_x = Array(Float64, n)
    ls_gradient = Array(Float64, n)

    # Show trace
    tr = OptimizationTrace()
    if store_trace || show_trace
        accelerated_gradient_descent_trace!(tr,
                                            x_t,
                                            f_x,
                                            gradient,
                                            iteration,
                                            store_trace,
                                            show_trace)
    end

    # Monitor convergence
    converged = false

    # Iterate until the norm of the gradient is within tolerance of zero
    while !converged && iteration <= iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Use a back-tracking line search to select a step-size
        #  Direction is always negative gradient
        for i in 1:n
            p[i] = -gradient[i]
        end
        alpha, f_update, g_update =
          line_search!(d, x_t, p, ls_x, ls_gradient, c1 = 0.5, rho = 0.99, alpha = 0.05)
        f_calls += f_update
        g_calls += g_update

        # Move in the direction of the gradient
        copy!(y_t1, y_t)
        for i in 1:n
            y_t[i] = x_t1[i] + alpha * p[i]
        end

        # x_t1 = x_t
        copy!(x_t1, x_t)

        # Take Nesterov step
        scaling = (iteration - 1.0) / (iteration + 2.0)
        for i in 1:n
            x_t[i] = y_t[i] + scaling * (y_t[i] - y_t1[i])
        end

        # Update current value of f and g
        f_x = d.fg!(x_t, gradient)
        f_calls += 1
        g_calls += 1

        # Assess convergence
        if norm(gradient, Inf) <= tolerance
            converged = true
            continue
        end

        # Show trace
        if store_trace || show_trace
            accelerated_gradient_descent_trace!(tr,
                                                x_t,
                                                f_x,
                                                gradient,
                                                iteration,
                                                store_trace,
                                                show_trace)
        end
    end

    OptimizationResults("Accelerated Gradient Descent",
                        initial_x,
                        x_t,
                        f_x,
                        iteration,
                        converged,
                        tr,
                        f_calls,
                        g_calls)
end
