function newton(d::TwiceDifferentiableFunction,
                initial_x::Vector,
                tolerance::Float64,
                max_iterations::Int64,
                store_trace::Bool,
                show_trace::Bool)

    # Maintain a record of the initial state
    x = copy(initial_x)

    # Count number of parameters
    n = length(x)

    # Cache the values of f, g and h
    f_x = d.f(x)
    gradient = Array(Float64, n)
    d.g!(x, gradient)
    H = Array(Float64, n, n)
    d.h!(x, H)

    # Don't run forever
    i = 0

    # Maintain a trace of the system
    tr = OptimizationTrace()
    if store_trace || show_trace
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

    # Track convergence
    converged = false

    # Determine direction of line search
    dx = -H \ gradient

    while !converged && i < max_iterations
        # Update the iteration counter
        i += 1

        # Select a step size
        step_size, f_up, g_up = backtracking_line_search(d.f, d.g!, x, dx)

        # Update our position
        x += step_size * dx

        # Cache values again
        f_x = d.f(x)
        d.g!(x, gradient)
        d.h!(x, H)

        # Show state of the system
        if store_trace || show_trace
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

        # Select a search direction
        dx = -H \ gradient

        # Assess convergence
        l2 = dot(gradient, -dx)
        if l2 / 2 <= tolerance
           converged = true
        end
    end

    OptimizationResults("Newton's Method",
                        initial_x,
                        x,
                        f_x,
                        i,
                        converged,
                        tr)
end

function newton(d::TwiceDifferentiableFunction,
                initial_x::Vector)
    newton(d, initial_x, 1e-16, 1_000, false, false)
end

function newton(f::Function,
                g!::Function,
                h!::Function,
                initial_x::Vector,
                tolerance::Float64,
                max_iterations::Int64,
                store_trace::Bool,
                show_trace::Bool)
    d = TwiceDifferentiableFunction(f, g!, h!)
    newton(d,
           initial_x,
           tolerance,
           max_iterations,
           store_trace,
           show_trace)
end

function newton(f::Function,
                g!::Function,
                h!::Function,
                initial_x::Vector)
    d = TwiceDifferentiableFunction(f, g!, h!)
    newton(d, initial_x, 1e-16, 1_000, false, false)
end
