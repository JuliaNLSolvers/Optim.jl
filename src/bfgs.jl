function bfgs(d::DifferentiableFunction,
              initial_x::Vector,
              initial_B::Matrix,
              tolerance::Float64,
              max_iterations::Integer,
              store_trace::Bool,
              show_trace::Bool)

    # Keep track of the number of iterations
    k = 0

    # Keep a record of our current position
    x_old = copy(initial_x)
    x_new = copy(initial_x)

    # Determine number of parameters
    n = length(initial_x)
    gradient_old = Array(Float64, n)
    gradient_new = Array(Float64, n)

    # Keep a record of the current gradient
    f_x = f(x)
    d.g!(x_old, gradient_old)
    d.g!(x_new, gradient_new)

    # Initialize our approximate inverse Hessian
    B = copy(initial_B)

    # Iterate until convergence
    converged = false

    # Show state of the system
    tr = OptimizationTrace()
    if store_trace || show_trace
        dt = Dict()
        dt["g(x_new)"] = copy(gradient_new)
        dt["~inv(h)"] = copy(B)
        os = OptimizationState(x_old, f_x, k, dt)
        if store_trace
            push!(tr, os)
        end
        if show_trace
            println(os)
        end
    end

    # Maintain arrays for position and gradient changes
    s = Array(Float64, n)
    y = Array(Float64, n)

    while !converged && k < max_iterations
        # Increment the iteration counter
        k += 1

        # Set the search direction
        p = -B * gradient_new

        # Calculate a step-size
        # TODO: Clean up backtracking_line_search
        alpha, f_update, g_update = backtracking_line_search(d.f, d.g!, x_new, p)

        # Update our position
        s = alpha * p
        copy!(x_old, x_new)
        x_new = x_old + s

        # Update the gradient
        copy!(gradient_old, gradient_new)
        g!(x_new, gradient_new)

        # Update the change in the gradient
        for index in 1:n
            y[index] = gradient_new[index] - gradient_old[index]
        end

        # Update the inverse Hessian approximation
        rho = 1.0 / dot(y, s)
        if rho == Inf
           println("Cannot decrease the objective function along the current search direction")
           break
        end
        # MAINTAIN ACROSS CALLS
        v = eye(size(h, 1)) - rho * y * s'
        if k == 1
          h = v'v * dot(y, s) / dot(y, y)
        else
          h = v' * h * v + rho * s * s'
        end

        # Show state of the system
        if store_trace || show_trace
            d = Dict()
            d["g(x_new)"] = copy(gradient_new)
            d["~inv(H)"] = copy(B)
            d["Step-size"] = alpha
            d["First-order opt."] = norm(gradient_new, Inf)
            os = OptimizationState(x_new, f(x_new), k, d)
            if store_trace
                push!(tr, os)
            end
            if show_trace
                println(os)
            end
        end

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
