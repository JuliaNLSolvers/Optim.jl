function naive_gradient_descent_trace!(tr::OptimizationTrace,
                                       x::Vector,
                                       f_x::Real,
                                       gradient::Vector,
                                       iteration::Integer,
                                       store_trace::Bool,
                                       show_trace::Bool)
    dt = Dict()
    dt["g(x)"] = copy(gradient)
    dt["|g(x)|"] = norm(gradient)
    os = OptimizationState(copy(x), f_x, iteration, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function naive_gradient_descent(d::DifferentiableFunction,
                                initial_x::Vector;
                                step_size::Real = 0.1,
                                tolerance::Real = 1e-8,
                                iterations::Integer = 1_000,
                                store_trace::Bool = false,
                                show_trace::Bool = false)

    # Keep a copy of the initial state of the system
    x = copy(initial_x)

    # Keep track of the number of iterations
    iteration = 0

    # Track calls to f, g! and fg!
    f_calls, g_calls = 0, 0

    # Allocate vectors for re-use by gradient calls
    n = length(x)
    gradient = Array(Float64, n)

    # Maintain current value of f and g
    f_calls += 1
    g_calls += 1
    f_x = d.fg!(x, gradient)

    # Track convergence
    converged = false

    # Maintain trace information
    tr = OptimizationTrace()
    if store_trace || show_trace
        naive_gradient_descent_trace!(tr, x, f_x,
                                      gradient, iteration,
                                      store_trace, show_trace)
    end

    # Iterate until convergence
    while !converged && iteration < iterations
        # Update the iteration counter
        iteration += 1

        # Set local step size
        alpha = step_size / iteration

        # Update our position
        for i in 1:n
            x[i] = x[i] - alpha * gradient[i]
        end

        # Update the cached values of f and g
        f_calls += 1
        g_calls += 1
        f_x = d.fg!(x, gradient)

        # Assess convergence
        if norm(gradient, Inf) <= tolerance
            converged = true
        end

        # Update trace
        if store_trace || show_trace
            naive_gradient_descent_trace!(tr, x, f_x,
                                          gradient, iteration,
                                          store_trace, show_trace)
        end
    end

    OptimizationResults("Constant Step-Size Gradient Descent",
                        initial_x,
                        x,
                        f_x,
                        iteration,
                        converged,
                        tr,
                        f_calls,
                        g_calls)
end
