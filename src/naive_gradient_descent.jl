function naive_gradient_descent(d::DifferentiableFunction,
                                initial_x::Vector,
                                step_size::Float64,
                                tolerance::Float64,
                                max_iterations::Integer,
                                store_trace::Bool,
                                show_trace::Bool)

    # Keep a copy of the initial state of the system
    x = copy(initial_x)

    # Keep track of the number of iterations
    i = 0

    # Track calls to f and g!
    f_calls, g_calls = 0, 0

    # Allocate vectors for re-use by gradient calls
    n = length(x)
    gradient = Array(Float64, n)

    # Maintain current value of f and g
    f_calls += 1
    g_calls += 1
    f_x = d.f(x)
    d.g!(x, gradient)

    # Track convergence
    converged = false

    # Maintain trace information
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

    # Iterate until convergence
    while !converged && i < max_iterations
        # Update the iteration counter
        i = i + 1

        # Update our position
        x = x - step_size * gradient

        # Update the cached values of f and g
        f_calls += 1
        g_calls += 1
        f_x = d.f(x)
        d.g!(x, gradient)

        # Assess convergence
        if norm(gradient) <= tolerance
            converged = true
        end

        # Update trace
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

    OptimizationResults("Constant Step-Size Gradient Descent",
                        initial_x,
                        x,
                        f_x,
                        i,
                        converged,
                        tr)
end

function naive_gradient_descent(d::DifferentiableFunction,
                                initial_x::Vector,
                                step_size::Float64)
    naive_gradient_descent(d,
                           initial_x,
                           step_size,
                           1e-8,
                           1_000,
                           false,
                           false)
end

function naive_gradient_descent(d::DifferentiableFunction,
                                initial_x::Vector)
    naive_gradient_descent(d,
                           initial_x,
                           0.01,
                           1e-8,
                           1_000,
                           false,
                           false)
end

function naive_gradient_descent(f::Function,
                                g!::Function,
                                initial_x::Vector,
                                step_size::Float64,
                                tolerance::Float64,
                                max_iterations::Integer,
                                store_trace::Bool,
                                show_trace::Bool)
    d = DifferentiableFunction(f, g!)
    naive_gradient_descent(d,
                           initial_x,
                           step_size,
                           tolerance,
                           max_iterations,
                           store_trace,
                           show_trace)
end

function naive_gradient_descent(f::Function,
                                g!::Function,
                                initial_x::Vector,
                                step_size::Float64)
    d = DifferentiableFunction(f, g!)
    naive_gradient_descent(d,
                           initial_x,
                           step_size,
                           1e-8,
                           1_000,
                           false,
                           false)
end

function naive_gradient_descent(f::Function,
                                g!::Function,
                                initial_x::Vector)
    d = DifferentiableFunction(f, g!)
    naive_gradient_descent(d,
                           initial_x,
                           0.1,
                           1e-8,
                           1_000,
                           false,
                           false)
end
