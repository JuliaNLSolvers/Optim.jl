function newton(f::Function,
                g::Function,
                h::Function,
                initial_x::Vector,
                tolerance::Float64,
                max_iterations::Int64,
                store_trace::Bool,
                show_trace::Bool)

    # Maintain a record of the state.
    x = initial_x

    # Don't run forever.
    i = 0

    # Maintain a trace of the system.
    tr = OptimizationTrace()
    if store_trace || show_trace
        d = Dict()
        d["g(x)"] = g(x)
        d["h(x)"] = h(x)
        os = OptimizationState(x, f(x), i, d)
        if store_trace
            push!(tr, os)
        end
        if show_trace
            println(os)
        end
    end

    # Track convergence.
    converged = false

    # Select a stepsize.
    dx = -h(x) \ g(x)
    l2 = dot(g(x), -dx)

    while !converged && i < max_iterations
        # Update the iteration counter.
        i += 1

        # Select a step size.
        step_size = backtracking_line_search(f, g, x, dx)

        # Update our position.
        x += step_size * dx

        # Show state of the system.
        if store_trace || show_trace
            d = Dict()
            d["g(x)"] = g(x)
            d["h(x)"] = h(x)
            os = OptimizationState(x, f(x), i, d)
            if store_trace
                push!(tr, os)
            end
            if show_trace
                println(os)
            end
        end

        # Select a search direction.
        dx = -h(x) \ g(x)

        # Assess convergence.
        l2 = dot(g(x), -dx)
        if l2 / 2 <= tolerance
           converged = true
        end
    end

    OptimizationResults("Newton's Method",
                        initial_x,
                        x,
                        f(x),
                        i,
                        converged,
                        tr)
end

function newton(f::Function, g::Function, h::Function, initial_x::Vector,
                store_trace::Bool)
    newton(f, g, h, initial_x, 10e-16, 1000, store_trace, false)
end
function newton(f::Function, g::Function, h::Function, initial_x::Vector)
    newton(f, g, h, initial_x, 10e-16, 1_000, false, false)
end
