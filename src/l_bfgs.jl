# Notational note
# JMW's dx_history <=> NW's S
# JMW's dgr_history <=> NW's Y

# Here alpha is a cache that parallels betas
# It is not the step-size
# q is also a cache
function twoloop!(s::Vector,
                  gr::Vector,
                  rho::Vector,
                  dx_history::Matrix,
                  dgr_history::Matrix,
                  m::Integer,
                  iteration::Integer,
                  alpha::Vector,
                  q::Vector)
    # Count number of parameters
    n = length(s)

    # Determine lower and upper bounds for loops
    lower = iteration - m
    upper = iteration - 1

    # Copy gr into q for backward pass
    copy!(q, gr)

    # Backward pass
    for index in upper:-1:lower
        if index < 1
            continue
        end
        i = mod1(index, m)
        alpha[i] = rho[i] * dot(dx_history[:, i], q)
        for j in 1:n
            q[j] -= alpha[i] * dgr_history[j, i]
        end
    end

    # Copy q into s for forward pass
    copy!(s, q)

    # Forward pass
    for index in lower:1:upper
        if index < 1
            continue
        end
        i = mod1(index, m)
        beta = rho[i] * dot(dgr_history[:, i], s)
        for j in 1:n
            s[j] += dx_history[j, i] * (alpha[i] - beta)
        end
    end

    # Negate search direction
    for i in 1:n
        s[i] = -1.0 * s[i]
    end

    return
end

macro lbfgstrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
                dt["Current step size"] = alpha
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

function l_bfgs{T}(d::Union(DifferentiableFunction,
                            TwiceDifferentiableFunction),
                   initial_x::Vector{T};
                   m::Integer = 10,
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

    # Store a history of changes in position and gradient
    rho = Array(Float64, m)
    dx_history = Array(Float64, n, m)
    dgr_history = Array(Float64, n, m)

    # The current search direction
    s = Array(Float64, n)

    # Buffers for use in line search
    x_ls = Array(Float64, n)
    gr_ls = Array(Float64, n)

    # Store f(x) in f_x
    f_x_previous = NaN
    f_x = d.fg!(x, gr)
    f_calls += 1
    g_calls += 1
    copy!(gr_previous, gr)

    # Keep track of step-sizes
    alpha = alphainit(1.0, x, gr, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Buffers for new entries of dx_history and dgr_history
    dx = Array(Float64, n)
    dgr = Array(Float64, n)

    # Buffers for use by twoloop!
    twoloop_q = Array(Float64, n)
    twoloop_alpha = Array(Float64, m)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace
    @lbfgstrace

    # Iterate until convergence
    x_converged = false
    f_converged = false
    gr_converged = false
    converged = false
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Determine the L-BFGS search direction
        twoloop!(s, gr, rho, dx_history, dgr_history, m, iteration,
                 twoloop_alpha, twoloop_q)

        # Refresh the line search cache
        dphi0 = dot(gr, s)
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

        # Update the L-BFGS history of positions and gradients
        rho_iteration = 1.0 / dot(dx, dgr)
        if isinf(rho_iteration)
            # TODO: Introduce a formal error? There was a warning here previously
            break
        end
        dx_history[:, mod1(iteration, m)] = dx
        dgr_history[:, mod1(iteration, m)] = dgr
        rho[mod1(iteration, m)] = rho_iteration

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

        @lbfgstrace
    end

    MultivariateOptimizationResults("L-BFGS",
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
