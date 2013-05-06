# Translation from our variables to Nocedal and Wright's
# JMW's dx <=> NW's s
# JMW's dgr <=> NW' y

function bfgs_trace!(tr::OptimizationTrace,
                     x::Vector,
                     f_x::Real,
                     gr::Vector,
                     invH::Matrix,
                     iteration::Integer,
                     store_trace::Bool,
                     show_trace::Bool)
    dt = Dict()
    dt["g(x)"] = copy(gr)
    dt["Maximum component of g(x)"] = norm(gr, Inf)
    dt["~inv(H)"] = copy(invH)
    os = OptimizationState(copy(x), f_x, iteration, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function bfgs{T}(d::Union(DifferentiableFunction,
                          TwiceDifferentiableFunction),
                 initial_x::Vector{T};
                 initial_invH::Matrix = eye(length(initial_x)),
                 xtol::Real = 1e-32,
                 ftol::Real = 1e-32,
                 grtol::Real = 1e-8,
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 linesearch!::Function = hz_linesearch!)

    # Maintain current state in x and previous state in x_previous
    x = copy(initial_x)
    x_previous = copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls = 0
    g_calls = 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr and previous gradient in gr_previous
    gr = Array(Float64, n)
    gr_previous = Array(Float64, n)

    # Store the approximate inverse Hessian in invH
    invH = copy(initial_invH)

    # The current search direction
    s = Array(Float64, n)

    # For something????
    u = Array(Float64, n)

    # Buffers for use in line search
    x_ls = Array(Float64, n)
    gr_ls = Array(Float64, n)

    # Store f(x) in f_x
    f_x = d.fg!(x, gr)
    f_calls += 1
    g_calls += 1
    copy!(gr_previous, gr)

    # Store the history of function values
    f_values = Array(T, iterations + 1)
    fill!(f_values, nan(T))
    f_values[iteration + 1] = f_x

    # Keep track of step-sizes
    alpha = alphainit(1.0, x, gr, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Maintain record of changes in position and gradient
    dx = Array(Float64, n)
    dgr = Array(Float64, n)

    # Maintain a cached copy of the identity matrix
    I = eye(size(invH)...)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace
    if tracing
        bfgs_trace!(tr, x, f_x, gr, invH,
                    iteration, store_trace, show_trace)
    end

    # Iterate until convergence
    x_converged = false
    f_converged = false
    gr_converged = false
    converged = false
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Set the search direction        
        # Search direction is the negative gradient divided by the approximate Hessian
        A_mul_B(s, invH, gr)
        for i in 1:n
            s[i] = -s[i]
        end

        # Refresh the line search cache
        dphi0 = dot(gr, s)
        # If invH is not positive definite, reset it to I
        if dphi0 > 0.0
            copy!(invH, I)
            s = -gr
            dphi0 = dot(gr, s)
        end
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          linesearch!(d, x, s, x_ls, gr_ls, lsr, alpha, mayterminate)
        f_calls += f_update
        g_calls += g_update

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position
        for i in 1:n
            dx[i] = alpha * s[i]
            x[i] = x[i] + dx[i]
        end

        # Maintain a record of the previous gradient
        copy!(gr_previous, gr)

        # Update the function value and gradient
        f_x = d.fg!(x, gr)
        f_calls += 1
        g_calls += 1
        f_values[iteration + 1] = f_x

        # Measure the change in the gradient
        for i in 1:n
            dgr[i] = gr[i] - gr_previous[i]
        end

        # Update the inverse Hessian approximation using Sherman-Morrison
        dx_dgr = dot(dx, dgr)
        if dx_dgr == 0.0
            break
        end
        A_mul_B(u, invH, dgr)

        c1 = (dx_dgr + dot(dgr, u)) / (dx_dgr * dx_dgr)
        c2 = 1 / dx_dgr

        # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
        for i in 1:n
            for j in 1:n
                invH[i, j] += c1 * dx[i] * dx[j] - c2 * (u[i] * dx[j] + u[j] * dx[i])
            end
        end

        # Assess convergence
        deltax = 0.0
        for i in 1:n
            diff = abs(x[i] - x_previous[i])
            if diff > deltax
                deltax = diff
            end
        end
        if deltax < xtol
            x_converged = true
        end
        if abs(f_values[iteration + 1] - f_values[iteration]) < ftol
            f_converged = true
        end
        if norm(gr, Inf) < grtol
            gr_converged = true
        end
        converged = x_converged || f_converged || gr_converged

        # Show trace
        if tracing
            bfgs_trace!(tr, x, f_x, gr, invH,
                        iteration, store_trace, show_trace)
        end
    end

    OptimizationResults("BFGS",
                        initial_x,
                        x,
                        f_x,
                        iteration,
                        iteration == iterations,
                        x_converged,
                        xtol,
                        f_converged,
                        ftol,
                        gr_converged,
                        grtol,
                        tr,
                        f_calls,
                        g_calls,
                        f_values[1:(iteration + 1)])
end
