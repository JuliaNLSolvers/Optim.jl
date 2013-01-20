function bfgs(f::Function,
              g::Function,
              initial_x::Vector,
              initial_h::Matrix,
              tolerance::Float64,
              max_iterations::Int64,
              store_trace::Bool,
              show_trace::Bool)

    # Keep track of the number of iterations.
    k = 0

    # Keep a record of our current position.
    x_new = initial_x
    x_old = initial_x

    # Keep a record of the current gradient.
    gradient_new = g(x_new)
    gradient_old = g(x_old)

    # Initialize our approximate Hessian.
    h = initial_h

    # Iterate until convergence.
    converged = false

    # Show state of the system.
    tr = OptimizationTrace()
    if store_trace || show_trace
        d = Dict()
        d["g(x_new)"] = g(x_new)
        d["h"] = h
        os = OptimizationState(x_new, f(x_new), k, d)
        if store_trace
            push!(tr, os)
        end
        if show_trace
            println(os)
        end
    end

    while !converged && k < max_iterations
        # Increment the iteration counter.
        k += 1

        # Set the search direction.
        p = -h * gradient_new

        # Calculate a step-size.
        alpha = backtracking_line_search(f, g, x_new, p)

        # Show state of the system.
        if store_trace || show_trace
            d = Dict()
            d["g(x_new)"] = g(x_new)
            d["h"] = h
            d["Step-size"] = alpha
            d["First-order opt."] = norm(gradient_old, Inf)
            os = OptimizationState(x_new, f(x_new), k, d)
            if store_trace
                push!(tr, os)
            end
            if show_trace
                println(os)
            end
        end

        # Update our position.
        x_old = x_new
        x_new = x_old + alpha * p
        s = x_new - x_old

        # Update the gradient.
        gradient_old = gradient_new
        gradient_new = g(x_new)
        y = gradient_new - gradient_old

        # Update the Hessian.
        rho = 1.0 / dot(y, s)
        if rho == Inf
           println("Cannot decrease the objective function along the current search direction")
           break
        end
        v = eye(size(h, 1)) - rho * y * s'
        h = (k == 1 ? v'v * dot(y, s) / dot(y, y) : v' * h * v + rho * s * s')

        # Assess convergence.
        if norm(gradient_new, Inf) <= tolerance
           converged = true
        end
    end

    OptimizationResults("BFGS", initial_x, x_new, f(x_new), k, converged, tr)
end

bfgs(f::Function,
     g::Function,
     initial_x::Vector,
     initial_h::Matrix,
     store_trace::Bool,
     show_trace::Bool) = bfgs(f, g, initial_x, initial_h, 10e-8, 1_000, store_trace, show_trace)

bfgs(f::Function,
     g::Function,
     initial_x::Vector,
     initial_h::Matrix) = bfgs(f, g, initial_x, initial_h, false, false)

bfgs(f::Function,
     g::Function,
     initial_x::Vector,
     store_trace::Bool,
     show_trace::Bool) = bfgs(f, g, initial_x, eye(length(initial_x)), 10e-8, 1_000, store_trace, show_trace)

bfgs(f::Function,
     g::Function,
     initial_x::Vector) = bfgs(f, g, initial_x, eye(length(initial_x)), 10e-8, 1_000, false, false)
