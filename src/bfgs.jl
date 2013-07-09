# Translation from our variables to Nocedal and Wright's
# JMW's dx <=> NW's s
# JMW's dgr <=> NW' y

macro bfgstrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
                dt["~inv(H)"] = copy(invH)
            end
            grnorm = norm(gr, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
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
                 extended_trace::Bool = false,
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
    f_x_previous = NaN
    f_calls += 1
    g_calls += 1
    copy!(gr_previous, gr)

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
    @bfgstrace

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
        f_x_previous = f_x
        f_x = d.fg!(x, gr)
        f_calls += 1
        g_calls += 1

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

        x_converged,
        f_converged,
        gr_converged,
        converged = assess_convergence(x,
                                       x_previous,
                                       f_x,
                                       f_x_previous,
                                       gr,
                                       xtol,
                                       ftol,
                                       grtol)

        @bfgstrace
    end

    MultivariateOptimizationResults("BFGS",
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
                        g_calls)
end
