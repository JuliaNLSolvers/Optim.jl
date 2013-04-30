function bfgs_trace!(tr::OptimizationTrace,
                     x_old::Vector,
                     f_x::Real,
                     iteration::Integer,
                     gradient_new::Vector,
                     B::Matrix,
                     store_trace::Bool,
                     show_trace::Bool)
    dt = Dict()
    dt["g(x_new)"] = copy(gradient_new)
    dt["~inv(H)"] = copy(B)
    os = OptimizationState(copy(x_old), f_x, iteration, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function bfgs(d::DifferentiableFunction,
              initial_x::Vector;
              initial_B::Matrix = eye(length(initial_x)),
              tolerance::Real = 1e-8,
              iterations::Integer = 1_000,
              store_trace::Bool = false,
              show_trace::Bool = false,
              line_search!::Function = interpolating_line_search!)

    # Keep track of the number of iterations
    iteration = 0

    # Count function and gradient calls
    f_calls = 0
    g_calls = 0

    # Keep a record of our current position
    x_old = copy(initial_x)
    x_new = copy(initial_x)

    # Determine number of parameters
    n = length(initial_x)
    gradient_old = Array(Float64, n)
    gradient_new = Array(Float64, n)

    # Keep a record of the current gradient
    f_calls += 1
    g_calls += 1
    f_x = d.fg!(x_old, gradient_old)
    copy!(gradient_new, gradient_old)

    # Initialize our approximate inverse Hessian
    B = copy(initial_B)

    # Iterate until convergence
    converged = false

    # Show state of the system
    tr = OptimizationTrace()
    if store_trace || show_trace
        bfgs_trace!(tr,
                    x_old,
                    f_x,
                    iteration,
                    gradient_new,
                    B,
                    store_trace,
                    show_trace)
    end

    # Maintain arrays for position and gradient changes
    s = Array(Float64, n)
    y = Array(Float64, n)
    p = Array(Float64, n)
    u = Array(Float64, n)

    # Reuse arrays during every line search
    ls_x = Array(Float64, n)
    ls_gradient = Array(Float64, n)

    # Reuse identity matrix during BFGS approximate inverse Hessian updates
    I = eye(size(B, 1))

    while !converged && iteration < iterations
        # Increment the iteration counter
        iteration += 1

        # Set the search direction        
        A_mul_B(p, B, gradient_new)  # p = - B * gradient_new
        for i = 1 : n
            p[i] = -p[i]
        end

        # Calculate a step-size
        alpha, f_update, g_update =
          line_search!(d, x_new, p, ls_x, ls_gradient)
        f_calls += f_update
        g_calls += g_update

        # Update our position
        copy!(x_old, x_new)
        for i in 1:n
            s[i] = alpha * p[i]
            x_new[i] = x_old[i] + s[i]
        end

        # Update the gradient
        copy!(gradient_old, gradient_new)
        f_x = d.fg!(x_new, gradient_new)
        f_calls += 1
        g_calls += 1

        # Update the change in the gradient
        for i in 1:n
            y[i] = gradient_new[i] - gradient_old[i]
        end

        # Update the inverse Hessian approximation
        # (using the formula in Wikipedia)
        
        sy = dot(s, y)
        if sy == 0
            break
        end
        A_mul_B(u, B, y)  # u = B * y
        
        c1 = (sy + dot(y, u)) / (sy * sy)
        c2 = 1 / sy
        
        # B = B + c1 * (s * s') - c2 * (u * s' + s * u')                
        for i = 1 : n, j = 1 : n
        	B[i,j] += c1 * s[i] * s[j] - c2 * (u[i] * s[j] + u[j] * s[i])
        end

        # Show state of the system
        if store_trace || show_trace
            bfgs_trace!(tr,
                        x_old,
                        f_x,
                        iteration,
                        gradient_new,
                        B,
                        store_trace,
                        show_trace)
        end

        # Assess convergence
        if norm(gradient_new, Inf) <= tolerance
           converged = true
        end
    end

    OptimizationResults("BFGS",
                        initial_x,
                        x_new,
                        f_x,
                        iteration,
                        converged,
                        tr,
                        f_calls,
                        g_calls)
end
