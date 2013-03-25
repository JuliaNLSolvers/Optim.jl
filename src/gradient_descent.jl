function gradient_descent(d::DifferentiableFunction,
                          initial_x::Vector,
                          tolerance::Float64,
                          max_iterations::Integer,
                          store_trace::Bool,
                          show_trace::Bool)

    # Keep a copy of the initial state of the system
    x = copy(initial_x)

    # Count the number of gradient descent steps we perform
    i = 0

    # Track calls to f and g!
    f_calls, g_calls = 0, 0

    # Allocate vectors for re-use by gradient calls
    n = length(x)
    gradient = Array(Float64, n)
    p = Array(Float64, n)

    # Maintain current value of f and g
    f_calls += 1
    g_calls += 1
    f_x = d.f(x)
    d.g!(x, gradient)

    # Show trace
    tr = OptimizationTrace()
    if store_trace || show_trace
        dt = Dict()
        dt["g(x)"] = copy(gradient)
        dt["|g(x)|"] = norm(gradient)
        os = OptimizationState(x, f_x, i, dt)
        if store_trace
            push!(tr, os)
        end
        if show_trace
            println(os)
        end
    end

    # Monitor convergence
    converged = false

    # Iterate until the norm of the gradient is within tolerance of zero
    while !converged && i < max_iterations
        # Increment the number of steps we've had to perform
        i = i + 1

        # Use a back-tracking line search to select a step-size
        # TODO: Return f_calls, g_calls from backtracking
        # TODO: Reuse the p vector
        for index in 1:n
            p[index] = -gradient[index]
        end
        step_size, f_update, g_update =
          backtracking_line_search(d.f, d.g!, x, p)
        f_calls += f_update
        g_calls += g_update

        # Move in the direction of the gradient
        x = x - step_size * gradient

        # Update the function value and gradient
        f_calls += 1
        g_calls += 1
        f_x = d.f(x)
        d.g!(x, gradient)

        # Assess convergence
        if norm(gradient) <= tolerance
           converged = true
        end

        # Show trace.
        if store_trace || show_trace
            dt = Dict()
            dt["g(x)"] = copy(gradient)
            dt["|g(x)|"] = norm(gradient)
            os = OptimizationState(x, f_x, i, dt)
            if store_trace
                push!(tr, os)
            end
            if show_trace
                println(os)
            end
        end
    end

    OptimizationResults("Gradient Descent w/ Backtracking Line Search",
                        initial_x,
                        x,
                        f_x,
                        i,
                        converged,
                        tr)
end

function gradient_descent(d::DifferentiableFunction,
                          initial_x::Vector)
    gradient_descent(d, initial_x, 1e-8, 1_000, false, false)
end

function gradient_descent(f::Function,
                          g!::Function,
                          initial_x::Vector,
                          tolerance::Float64,
                          max_iterations::Integer,
                          store_trace::Bool,
                          show_trace::Bool)
  d = DifferentiableFunction(f, g!)
  return gradient_descent(d,
                          initial_x,
                          tolerance,
                          max_iterations,
                          store_trace,
                          show_trace)
end

function gradient_descent(f::Function,
                          initial_x::Vector,
                          tolerance::Float64,
                          max_iterations::Integer,
                          store_trace::Bool,
                          show_trace::Bool)
    d = DifferentiableFunction(f)
    return gradient_descent(d,
                            initial_x,
                            tolerance,
                            max_iterations,
                            store_trace,
                            show_trace)
end

function gradient_descent(f::Function,
                          g!::Function,
                          initial_x::Vector)
    gradient_descent(f, g!, initial_x, 1e-8, 1_000, false, false)
end

function gradient_descent(f::Function,
                          initial_x::Vector)
    gradient_descent(f, initial_x, 1e-8, 1_000, false, false)
end
